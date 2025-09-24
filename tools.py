from langgraph import Tool
from typing import Optional, Dict, Any, List, Union
from datetime import time
from utils import log_time, process_pdfs, format_docs, format_chat_history_for_prompt, count_tokens
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, HumanMessage, AIMessage
from langchain.schema.runnable import RunnablePassthrough
import time
from config import (
    main_model,
    RAG_MODEL,
    PDF_STORAGE_PATH
)


@Tool
def hint_tool(language: str, problem_description: str) -> Optional[Dict[str, Any]]:
    """
    Generate step-by-step hints for solving a DSA problem
    """
    start = time.time()
    try:
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
        **Hint 3:** [hint text]
        etc.
        """
        
        hint_prompt = ChatPromptTemplate.from_template(hint_prompt_template)
        
        # Run the hint generation
        hint_chain = hint_prompt | main_model | StrOutputParser()
        hints_response = hint_chain.invoke({
            "language": language,
            "problem_description": problem_description
        })
        
        print(f"Generated hints: {hints_response}")
        
        log_time("Hint Generation", start)
        
        # Return the results
        return {
            "hints": hints_response,
            "error": None
        }
        
    except Exception as e:
        log_time("Hint Tool (Error)", start)
        print(f"Error generating hints: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "hints": None,
            "error": str(e)
        }


@Tool
def rag_tool(message: str, chat_history: List[Union[HumanMessage, AIMessage]]) -> Dict[str, Any]:
    """
    Search DSA knowledge base and generate a response.
    
    Args:
        message: The user's current message
        chat_history: List of previous chat messages
        
    Returns:
        Dict containing:
        - response: The generated response
        - chat_history: Updated chat history
        - token_usage: Token usage information
        - error: Error message if any
    """
    start = time.time()
    print("\n=== RAG TOOL STARTED ===")
    print(f"Processing message: {message}")
    
    try:
        # Process PDFs and create vector store
        print("Processing PDFs and creating vector store...")
        doc_search = process_pdfs(PDF_STORAGE_PATH)
        
        # Configure retriever for better results
        retriever = doc_search.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        print("Vector store and retriever configured")

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
        response = runnable.invoke(message)
        
        # Update chat history
        updated_chat_history = chat_history + [
            HumanMessage(content=message),
            AIMessage(content=response)
        ]
        
        # Count tokens
        input_tokens = count_tokens(message, RAG_MODEL)
        output_tokens = count_tokens(response, RAG_MODEL)

        log_time("RAG Tool", start)
        
        # Return the results
        return {
            "response": response,
            "chat_history": updated_chat_history,
            "token_usage": {
                "model": RAG_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            "error": None
        }

    except Exception as e:
        # Handle exceptions
        log_time("RAG Tool (Error)", start)
        print(f"Error in RAG tool: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "response": None,
            "chat_history": chat_history,
            "token_usage": None,
            "error": str(e)
        }