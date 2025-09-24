from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
from langgraph.graph import START, END, StateGraph 
from datetime import time
import time
from langchain.prompts import ChatPromptTemplate
from utils import log_time, format_chat_history_for_prompt, count_tokens, format_docs, process_pdfs
from langchain.schema import StrOutputParser, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
import re
import os
from config import (
    main_model,
    extraction_model,
    RAG_MODEL,
    PDF_STORAGE_PATH
)


class UserState(BaseModel):
    """State management for the DSA tutor workflow."""
    message: str  # Current user message
    chat_history: List[Union[HumanMessage, AIMessage]] = []  # Conversation history
    missing_info: List[str] = []  # Missing information
    is_seeking_hints: bool = False  # Whether user wants hints for a problem
    is_seeking_explanation: bool = False  # Whether user is asking for concept explanation
    extracted_info: Dict[str, Optional[str]] = {}  # Extracted information
    response: Optional[str] = None  # The final response
    error: Optional[str] = None  # Error message if any
    status: str = "pending"  # Current status (pending, processing, completed, error)
    hints: Optional[List[Dict[str, Any]]] = None  # Hints generated
    rag_token_usage: Optional[Dict[str, Union[str, int]]] = None  # Token usage information
    followup_questions: List[str] = []  # Added for followup questions

    class Config:
        arbitrary_types_allowed = True


def analyze_user_intent(state: UserState):
    """
    Use the LLM to analyze user intent and extract information needed for hints or examples
    """
    start = time.time()
    print("\n=== ANALYZING USER INTENT ===")
    print(f"Current message: {state.message}")
    
    # Only use last 3 messages for brevity
    formatted_history = format_chat_history_for_prompt(state.chat_history[-3:])
    
    # Create an information extraction prompt
    extraction_prompt_template = """
    You are DSAGPT, an expert DSA information extractor.
    Analyze the 'Current User Message' in the context of the 'Conversation History'.

    Conversation History:
    {history}

    Current User Message: {message}

    Your job is to determine:
    
    1. Does the user currently want hints for solving a DSA problem? (true/false)
       - This should be true ONLY if the user is asking for hints or guidance to solve a specific problem
       - If the user is asking for full solutions or code, gently note that we provide hints to encourage learning
       - Example of hint requests (wants_hints: true):
         * "Give me hints to solve binary search problem"
         * "How to approach the two sum problem?"
         * "I need guidance on knapsack"
       - Examples of explanation requests (wants_hints: false):
         * "What is binary search?"
         * "Explain quicksort algorithm"
         * "Difference between stack and queue"
       - If the Current User Message primarily provides details (like problem description, language)
         AND the Conversation History shows the assistant previously asked for this information
         for hints, then 'wants_hints' should be true.
       - If the user explicitly asks for hints (e.g., "give hints", "how to approach"), 'wants_hints' is true.

    2. Extract any relevant details from the Current User Message:
       - Programming language (e.g., Python, Java, C++)
       - Problem description (brief summary)
       - Data structure involved (if mentioned)
       - Algorithm type (if mentioned)
       - Constraints (if mentioned, e.g., time complexity requirements)
    
    Validate the extracted programming language and other details:
    - Language should be one of common ones: Python, Java, C++, etc.
    - Problem description should be clear
    
    Return your analysis as a JSON object with the following structure:
    {{
      "wants_hints": true/false,
      "extracted_info": {{
        "language": "extracted programming language or null if not found",
        "problem_description": "extracted problem description or null if not found",
        "data_structure": "extracted data structure or null if not found",
        "algorithm_type": "extracted algorithm type or null if not found",
        "constraints": "extracted constraints or null if not found"
      }},
      "validation": {{
        "language_valid": true/false, // true if it's a supported language
        "problem_description_valid": true/false, // true if description is clear
        "all_valid": true/false // true if all required are valid
      }},
      "missing_info": ["list of missing items - can include 'language', 'problem_description' if they were not found or are invalid"],
      "intent_analysis": {{
        "is_seeking_explanation": true/false, // true if asking for concept explanation
        "is_seeking_hints": true/false, // true if asking for problem-solving hints
        "primary_intent": "explanation" | "hints" | "information",
        "confidence": "high" | "medium" | "low"
      }},
      "reasoning": "brief explanation of your analysis"
    }}
    
    Provide ONLY the JSON output, no other text.
    """
    extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
    
    # Run the extraction process
    extraction_chain = extraction_prompt | extraction_model | StrOutputParser()
    result_str = extraction_chain.invoke({
        "message": state.message,
        "history": formatted_history
    })
    log_time("Intent Analysis", start)
    
    # Parse the result
    cleaned_result_str = result_str.strip()
    if cleaned_result_str.startswith('```'):
        cleaned_result_str = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned_result_str)
        cleaned_result_str = re.sub(r'```$', '', cleaned_result_str)
    cleaned_result_str = cleaned_result_str.strip()
    
    intent_analysis = json.loads(cleaned_result_str)
    
    print("\n=== INTENT ANALYSIS RESULTS ===")
    print(f"Extracted Info: {json.dumps(intent_analysis.get('extracted_info', {}), indent=2)}")
    print(f"Missing Info: {intent_analysis.get('missing_info', [])}")
    print(f"Is Seeking Hints: {intent_analysis.get('intent_analysis', {}).get('is_seeking_hints', False)}")
    print(f"Is Seeking Explanation: {intent_analysis.get('intent_analysis', {}).get('is_seeking_explanation', False)}")
    print(f"Primary Intent: {intent_analysis.get('intent_analysis', {}).get('primary_intent', 'unknown')}")
    print(f"Confidence: {intent_analysis.get('intent_analysis', {}).get('confidence', 'unknown')}")
    print("==============================\n")
    
    # Update the state with the analysis results
    state.extracted_info = intent_analysis.get("extracted_info", {})
    state.missing_info = intent_analysis.get("missing_info", [])
    state.is_seeking_hints = intent_analysis.get("intent_analysis", {}).get("is_seeking_hints", False)
    state.is_seeking_explanation = intent_analysis.get("intent_analysis", {}).get("is_seeking_explanation", False)
    
    # Return the updated state
    return state


def hint_tool(state: UserState) -> UserState:
    """
    Generate hints for DSA problems using LLM
    """
    start = time.time()
    print("\n=== HINT TOOL STARTED ===")
    try:
        # Check for missing information first
        if len(state.missing_info) > 0:
            print(f"Missing information detected: {state.missing_info}")
            state.status = "missing_info"
            return state

        # Get required information from extracted_info
        language = state.extracted_info.get("language")
        problem_description = state.extracted_info.get("problem_description")

        if not all([language, problem_description]):
            state.error = "Missing required information for hints"
            state.status = "error"
            return state

        print(f"GENERATING HINTS for: language={language}, problem={problem_description}")
        
        # Create a prompt for generating hints
        hint_prompt_template = """
        You are DSAGPT, an expert DSA tutor. Provide step-by-step hints for solving the following problem without giving the full solution. Encourage thinking.

        Programming Language: {language}
        Problem Description: {problem_description}

        Generate 3-5 progressive hints that guide the user to solve it themselves.
        
        Format the response as:
        **Hint 1:** [hint text]
        **Hint 2:** [hint text]
        etc.
        """
        
        hint_prompt = ChatPromptTemplate.from_template(hint_prompt_template)
        
        # Run the hint generation
        hint_chain = hint_prompt | main_model | StrOutputParser()
        hints_response = hint_chain.invoke({
            "language": language,
            "problem_description": problem_description
        })
        
        log_time("Hint Generation", start)
        
        # Update state with the result
        state.hints = [{"hints": hints_response}]
        state.status = "completed"
        
        # Format the response
        formatted_hints = f"Here are some hints for your problem:\n{hints_response}"
        
        state.response = formatted_hints
        print("FINAL RESPONSE: ")
        print(state.response)
        
        return state
            
    except Exception as e:
        log_time("Hint Tool (Error)", start)
        print(f"Error generating hints: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update state with error
        state.error = "An unexpected error occurred while generating hints. Please try again later."
        state.status = "error"
        return state


def rag_tool(state: UserState):
    start = time.time()
    print("\n=== RAG TOOL STARTED ===")
    print(f"Processing message: {state.message}")
    try:
        # Process PDFs and create vector store (assume PDFs are DSA related now)
        print("Processing PDFs and creating vector store...")
        doc_search = process_pdfs(PDF_STORAGE_PATH)
        
        # Configure retriever for better results
        retriever = doc_search.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("Vector store and retriever configured")

        # Use state's chat history instead of Chainlit session
        chat_history = state.chat_history
        print(f"Using chat history with {len(chat_history)} messages")
        
        # Create prompt template for RAG
        prompt = ChatPromptTemplate.from_template("""You are DSAGPT, an expert DSA tutor specializing in Data Structures and Algorithms. Your goal is to provide helpful, accurate, and personalized guidance to users learning DSA topics.

        **Important:** Always use simple, clear, and conversational language. Avoid jargon and explain any technical terms in plain English. Assume the user has basic programming knowledge but explain DSA concepts from fundamentals.

        Follow these guidelines:
        - Present yourself as a knowledgeable, professional DSA tutor
        - Provide clear explanations of DSA concepts in simple words
        - When appropriate, compare different data structures or algorithms (e.g., array vs linked list, bubble sort vs quicksort)
        - Answer questions about time/space complexity, implementations, use cases, and optimizations
        - Provide balanced advice considering pros and cons of different approaches
        - Suggest relevant follow-up questions the user might want to ask
        - If resources mention specific examples, cite them with their source
        - Politely decline questions unrelated to DSA
        - If the user is asking for code examples or hints, inform them you'll need details like programming language
        - For problem-solving, provide hints step-by-step without giving full solutions to encourage learning

        Context from DSA resources:
        {context}

        Conversation History:
        {chat_history}

        Current Question: {question}

        Answer as DSAGPT: (Remember: Use simple, clear, and conversational language!)
        """)

        # Create the RAG chain
        runnable = (
            {
                "context": retriever | format_docs,
                "chat_history": lambda x: format_chat_history_for_prompt(chat_history),
                "question": RunnablePassthrough()
            }
            | prompt
            | main_model
            | StrOutputParser()
        )
        
        # Get response from the chain
        response = runnable.invoke(state.message)
        
        # Update chat history
        chat_history.append(HumanMessage(content=state.message))
        chat_history.append(AIMessage(content=response))
        
        # Count tokens
        input_tokens = count_tokens(state.message, RAG_MODEL)
        output_tokens = count_tokens(response, RAG_MODEL)

        # Update state with the response, chat history, token usage, and followup questions
        state.response = response
        state.chat_history = chat_history
        state.rag_token_usage = {
            "model": RAG_MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        state.followup_questions = []  # Placeholder, as generate_rag_followup_questions is not implemented

    except Exception as e:
        # Handle exceptions
        state.error = str(e)
        state.response = None
        state.rag_token_usage = None
        state.followup_questions = []  # Empty list for followup questions on error

    # Return the updated state
    return state


def should_get_hints(state: UserState) -> Dict:
    """Determine if we should get hints based on the state."""
    result = state.is_seeking_hints and not state.is_seeking_explanation
    print(f"\n=== HINT ROUTING DECISION ===")
    print(f"is_seeking_hints: {state.is_seeking_hints}")
    print(f"is_seeking_explanation: {state.is_seeking_explanation}")
    print(f"Should get hints: {result}")
    print("==============================\n")
    return {"state": state, "should_proceed": result}


def should_get_explanation(state: UserState) -> Dict:
    """Determine if we should get explanation based on the state."""
    result = state.is_seeking_explanation
    print(f"\n=== EXPLANATION ROUTING DECISION ===")
    print(f"is_seeking_explanation: {state.is_seeking_explanation}")
    print(f"Should get explanation: {result}")
    print("===============================\n")
    return {"state": state, "should_proceed": result}


def should_collect_missing_info(state: UserState) -> Dict:
    """Determine if we need to collect missing information."""
    result = len(state.missing_info) > 0
    print(f"\n=== MISSING INFO CHECK ===")
    print(f"Missing info: {state.missing_info}")
    print(f"Should collect missing info: {result}")
    print("==========================\n")
    return {"state": state, "should_proceed": result}


def should_proceed_to_hints(state: UserState) -> Dict:
    """Determine if we should proceed to getting hints."""
    result = state.is_seeking_hints and not state.is_seeking_explanation and len(state.missing_info) == 0
    print(f"\n=== PROCEED TO HINTS CHECK ===")
    print(f"is_seeking_hints: {state.is_seeking_hints}")
    print(f"is_seeking_explanation: {state.is_seeking_explanation}")
    print(f"missing_info: {state.missing_info}")
    print(f"Should proceed to hints: {result}")
    print("==============================\n")
    return {"state": state, "should_proceed": result}


# Wrapper functions for conditional routing
def route_to_hints(state: UserState) -> bool:
    """Wrapper to extract boolean from should_get_hints result."""
    result = should_get_hints(state)
    return result["should_proceed"]


def route_to_explanation(state: UserState) -> bool:
    """Wrapper to extract boolean from should_get_explanation result."""
    result = should_get_explanation(state)
    return result["should_proceed"]


def route_to_missing_info(state: UserState) -> bool:
    """Wrapper to extract boolean from should_collect_missing_info result."""
    result = should_collect_missing_info(state)
    return result["should_proceed"]


def handle_missing_info(state: UserState) -> UserState:
    """Handle missing information by generating a response."""
    start = time.time()
    print("\n=== HANDLING MISSING INFO ===")
    print(f"Missing info: {state.missing_info}")
    
    # Format the missing info into a natural language list
    missing_items = []
    for item in state.missing_info:
        if item == "language":
            missing_items.append("programming language")
        elif item == "problem_description":
            missing_items.append("problem description")
    
    missing_info_text = ", ".join(missing_items[:-1])
    if len(missing_items) > 1:
        missing_info_text += f" and {missing_items[-1]}"
    else:
        missing_info_text = missing_items[0]
    
    # Create a prompt for the missing info response
    missing_info_prompt = ChatPromptTemplate.from_template("""
    You are DSAGPT, a helpful DSA tutor. The user wants hints for a DSA problem 
    but is missing some required information. Create a friendly, concise response asking 
    for the following missing information: {missing_info}.
    
    For programming language, suggest common ones like Python, Java, C++.
    For problem description, ask for a clear statement of the problem.
    
    Always use simple, clear, and conversational language. Avoid jargon and explain any technical terms in plain English.
    
    Keep your response brief and friendly.
    """
    )
    
    # Generate the response
    missing_info_chain = missing_info_prompt | main_model | StrOutputParser()
    response = missing_info_chain.invoke({
        "missing_info": missing_info_text
    })
    
    # Update the state
    state.response = response
    state.status = "missing_info"
    
    log_time("Missing Info Response", start)
    return state


graph = StateGraph(UserState)

graph.add_node("hint_tool", hint_tool)
graph.add_node("rag", rag_tool)
graph.add_node("router", analyze_user_intent)
graph.add_node("handle_missing_info_node", handle_missing_info)
graph.add_node("should_get_explanation", should_get_explanation)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_to_hints,  # Use the wrapper function
    {
        True: "hint_tool",
        False: "should_get_explanation"
    }
)

# Add conditional edge for explanation routing
graph.add_conditional_edges(
    "should_get_explanation",
    route_to_explanation,  # Use the wrapper function
    {
        True: "rag",
        False: "rag"  # Default to RAG for general information if not specifically seeking explanation
    }
)

# Add conditional edge from hint_tool to handle_missing_info_node
graph.add_conditional_edges(
    "hint_tool",
    route_to_missing_info,  # Use the wrapper function
    {
        True: "handle_missing_info_node",
        False: END
    }
)

graph.add_edge("rag", END)
graph.add_edge("handle_missing_info_node", END)

chatbot_graph = graph.compile()