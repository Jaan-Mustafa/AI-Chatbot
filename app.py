from langchain.prompts import ChatPromptTemplate
from utils import run_diagnostics, log_time
from graph import UserState, HumanMessage, AIMessage, chatbot_graph
from typing import Optional, Dict, List, Union, Any, Tuple
from datetime import time
import time
import chainlit as cl
import asyncio
import re
from langchain.schema.output_parser import StrOutputParser
from database.db import SessionLocal
from database.models import User, ChatMessage
from sqlalchemy.sql import func
from utils import format_chat_history_for_prompt
from config import(
    followup_model,
    DATABASE_URL_BASE
)

# Constants for prompt limits
DEFAULT_PROMPT_LIMIT = 50
TEST_USER_PROMPT_LIMIT = 5000

@cl.data_layer
def get_data_layer():
    """
    Use SQLAlchemy data layer for chat history persistence.
    """
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
    return SQLAlchemyDataLayer(conninfo=DATABASE_URL_BASE)

# Google OAuth callback for Chainlit authentication
@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    """
    This callback is now simplified to only handle authentication.
    The user will be created in our custom DB on their first message.
    This prevents a race condition with Chainlit's data layer initialization.
    """
    email = raw_user_data.get("email")
    if not email:
        print("Error: No email found in user data for OAuth.")
        return None

    # We are only concerned with returning the user object for Chainlit here.
    # The database interaction is moved to on_message to avoid startup conflicts.
    return cl.User(
        identifier=email,
        metadata={
            "email": email,
            "provider": provider_id,
            "raw_user_data": raw_user_data  # Store raw data for later use
        }
    )

@cl.on_chat_start
async def on_chat_start():
    try:
        # Create a unique thread ID for this conversation
        thread_id = f"chat_{int(time.time())}"
        
        # Store the thread ID in the user's session
        cl.user_session.set("thread_id", thread_id)

        # Get user info if available
        user = cl.user_session.get("user")
        if not user:
            await cl.Message(content="Error: User not authenticated. Please log in again.").send()
            return

        # Initialize user in database if needed
        db = SessionLocal()
        try:
            db_user = db.query(User).filter(User.email == user.identifier).first()
            if not db_user:
                db_user = User(
                    email=user.identifier,
                    prompt_count=0,
                    is_active=True
                )
                db.add(db_user)
                db.commit()
        finally:
            db.close()

         # Store the graph in the user's session
        cl.user_session.set("graph", chatbot_graph)


        # Create a more detailed DSA tutor prompt
        template = """You are DSAGPT, an expert DSA tutor specializing in Data Structures and Algorithms. Your goal is to provide helpful, accurate, and personalized guidance to users learning DSA topics.

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

        Answer as DSAGPT: (Remember: Use simple, clear, and conversational language!)"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize the LLM with the system prompt
        system_message = prompt.format(context="", chat_history="", question="")
        cl.user_session.set("system_prompt", system_message)

        # Initialize empty chat history
        cl.user_session.set("chat_history", [])

        # Create welcome message with branding
        welcome_message = (
            "Welcome to **DSAGPT**!\n\n"
            "**How I can help you:**\n"
            "✓ Understand data structures like arrays, linked lists, trees, graphs\n"
            "✓ Learn algorithms for sorting, searching, dynamic programming, etc.\n"
            "✓ Get hints for solving DSA problems\n"
            "✓ See code examples in languages like Python, Java, C++\n"
            "✓ Analyze time and space complexity\n"
            "✓ Practice with explanations and step-by-step guidance\n\n"
            "What DSA topic or problem would you like to explore today?"
        )
        
                # Store the welcome message in the database
        db = SessionLocal()
        try:
            chat_message = ChatMessage(
                user_email=user.identifier,
                content=welcome_message,
                role="assistant",
                conversation_id=thread_id,
                message_metadata={
                    "source": "langgraph",
                    "timestamp": time.time(),
                    "type": "welcome"
                }
            )
            db.add(chat_message)
            db.commit()
        finally:
            db.close()
        
        await cl.Message(content=welcome_message).send()
    except Exception as e:
        print(f"Error in on_chat_start: {e}")
        await cl.Message(content="Error: Could not initialize chat session. Please try again.").send()



def generate_rag_followup_questions(content: str, chat_history: List[Union[HumanMessage, AIMessage]]) -> List[str]:
    """Generate follow-up questions for RAG responses."""
    # Check if the content already contains follow-up questions
    if "Suggested Follow-up Questions:" in content:
        return []
        
    start = time.time()
    formatted_history = format_chat_history_for_prompt(chat_history[-3:])

    followup_prompt = ChatPromptTemplate.from_template("""
    You are a helpful DSA tutor. Based on the previous response and conversation history,
    generate 3 natural follow-up questions that would help the user better understand DSA concepts or problems.

    Previous Response:
    {response}

    Conversation History:
    {history}

    Generate exactly 3 follow-up questions that:
    - Are relevant to the information just provided
    - Help clarify or expand on key points
    - Guide the user towards understanding DSA better
    - Are clear and conversational

    Format your response exactly like this:
    1. First question here?
    2. Second question here?
    3. Third question here?

    The questions should be complete sentences that can stand alone.
    """)

    chain = followup_prompt | followup_model | StrOutputParser()
    followup_questions = chain.invoke({
        "response": content,
        "history": formatted_history
    })
    log_time("RAG Follow-up LLM", start)

    # Extract questions into a list
    questions = []
    for line in followup_questions.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("```"):
            question = re.sub(r'^\d+\.\s*', '', line)
            if question:
                questions.append(question)
                print(f"DEBUG: Extracted RAG follow-up question: {question}")
    
    return questions


@cl.on_chat_end
async def on_chat_end():
    """Cleanup persistence connections when the chat ends."""
    # No need to cleanup here as we're using global instances
    pass

@cl.on_stop
async def on_stop():
    """Cleanup persistence connections when the server stops."""

@cl.on_message
async def on_message(message: cl.Message):
    # Get user email from session
    user_email = message.author if isinstance(message.author, str) else None
    if not user_email:
        await message.update(content="Error: User email not found. Please try logging in again.")
        return

    # Ensure user exists in database and check prompt limits
    user = ensure_user_in_db(user_email)
    can_proceed, limit_message = update_prompt_count(user_email)
    
    if not can_proceed:
        await message.update(content=limit_message)
        return

    # Show the limit message to the user
    await cl.Message(content=limit_message).send()

    # Get or create chat history
    chat_history = cl.user_session.get("chat_history", [])
    
    print(f"\n\n===== PROCESSING NEW MESSAGE: {message.content} =====\n")
    start_total = time.time()

    if message.content.strip() == "/diagnostics":
        return await run_diagnostics()
    
    # Get the graph from the user's session
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    user = cl.user_session.get("user")
        
    if not graph or not thread_id:
        await cl.Message(content="Error: Session not properly initialized. Please refresh the page.").send()
        return

    if not user:
        await cl.Message(content="Error: User not authenticated. Please log in again.").send()
        return
    
    # Get user info from the session
    user_id = user.identifier  # This is the email from OAuth
        
    # Store the user message in the database
    db = SessionLocal()
    try:
        # Store the message
        chat_message = ChatMessage(
            user_email=user_id,
            content=message.content,
            role="user",
            conversation_id=thread_id,
            message_metadata={
                "source": "langgraph",
                "timestamp": time.time()
            }
        )
        db.add(chat_message)
        db.commit()
    finally:
        db.close()

    # Configure the graph execution
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }

    # Create initial state with chat history from database
    state = UserState(
        message=message.content,
        chat_history=chat_history,
        status="pending"
    )

    try:
        # Create a single message for streaming updates
        status_msg = cl.Message(content="")
        await status_msg.send()

        # Update status for intent analysis
        status_msg.content = "Analyzing your request..."
        await status_msg.update()

        try:
            # Run the graph workflow
            result = await chatbot_graph.ainvoke(state)
            
            # Handle the result based on the path taken
            if result.get("error"):
                status_msg.content = f"I apologize, but I encountered an error: {result['error']}"
                await status_msg.update()
                
                # Store error in database
                db = SessionLocal()
                try:
                    chat_message = ChatMessage(
                        user_email=user_id,
                        content=status_msg.content,
                        role="assistant",
                        conversation_id=thread_id,
                        message_metadata={
                            "source": "langgraph",
                            "timestamp": time.time(),
                            "type": "error"
                        }
                    )
                    db.add(chat_message)
                    db.commit()
                finally:
                    db.close()
                return

            if result.get("status") == "missing_info":
                # For missing info, stream the response line by line
                main_response = result.get("response", "")
                lines = main_response.split('\n')
                current_content = ""
                
                for line in lines:
                    current_content += line + "\n"
                    status_msg.content = current_content
                    await status_msg.update()
                    await asyncio.sleep(0.1)  # Small delay between lines
                
                # Store response in database
                db = SessionLocal()
                try:
                    chat_message = ChatMessage(
                        user_email=user_id,
                        content=main_response,
                        role="assistant",
                        conversation_id=thread_id,
                        message_metadata={
                            "source": "langgraph",
                            "timestamp": time.time(),
                            "type": "missing_info"
                        }
                    )
                    db.add(chat_message)
                    db.commit()
                finally:
                    db.close()
                
            elif result.get("hints"):
                # Only show loading messages when we're actually providing hints
                hint_loading_messages = [
                    "Analyzing your DSA problem... This may take a moment. We're breaking down the problem to provide helpful hints.",
                    "Identifying key concepts...",
                    "Thinking about optimal approaches...",
                    "Generating step-by-step hints...",
                    "Ensuring hints encourage learning...",
                    "Almost there! Formatting your DSA hints..."
                ]
                
                # Show loading messages with delay
                for msg in hint_loading_messages:
                    status_msg.content = msg
                    await status_msg.update()
                    await asyncio.sleep(1.5)  # 1.5 second delay between messages
                
                # Send the already formatted response from the tool
                main_response = result.get("response", "")
                lines = main_response.split('\n')
                current_content = ""
                
                for line in lines:
                    current_content += line + "\n"
                    status_msg.content = current_content
                    await status_msg.update()
                    await asyncio.sleep(0.1)  # Small delay between lines
                
                # Generate follow-up questions only if they're not already in the response
                followup_questions = generate_rag_followup_questions(main_response, state.chat_history)
                
                # Add follow-up questions as clickable actions
                if followup_questions:
                    # Add the header as part of the message
                    status_msg.content = f"{current_content}\n\n**Suggested Follow-up Questions:**"
                    
                    # Create actions for each question
                    actions = []
                    for question in followup_questions:
                        actions.append(
                            cl.Action(
                                name="ask_followup",
                                label=f"+ {question}",
                                payload={"question": question}
                            )
                        )
                    status_msg.actions = actions
                    await status_msg.update()
                
                # Store response in database
                db = SessionLocal()
                try:
                    chat_message = ChatMessage(
                        user_email=user_id,
                        content=status_msg.content,
                        role="assistant",
                        conversation_id=thread_id,
                        message_metadata={
                            "source": "langgraph",
                            "timestamp": time.time(),
                            "type": "hints",
                            "token_usage": result.get("rag_token_usage")
                        }
                    )
                    db.add(chat_message)
                    db.commit()
                finally:
                    db.close()
                
                cl.user_session.set("last_response_was_hint", True)
            else:
                # RAG loading messages
                rag_loading_messages = [
                    "Searching through DSA knowledge base...",
                    "Analyzing relevant concepts and examples...",
                    "Retrieving the most accurate information for you...",
                    "Compiling comprehensive response based on DSA principles..."
                ]
                
                # Show loading messages with delay
                for msg in rag_loading_messages:
                    status_msg.content = msg
                    await status_msg.update()
                    await asyncio.sleep(1.2)  # 1.2 second delay between messages
                
                # Send RAG response
                main_response = result.get("response", "I apologize, but I couldn't generate a response at this time.")
                lines = main_response.split('\n')
                current_content = ""
                
                for line in lines:
                    current_content += line + "\n"
                    status_msg.content = current_content
                    await status_msg.update()
                    await asyncio.sleep(0.1)  # Small delay between lines
                
                # Generate follow-up questions only if they're not already in the response
                followup_questions = generate_rag_followup_questions(main_response, state.chat_history)
                
                # Add follow-up questions as clickable actions
                if followup_questions:
                    # Add the header as part of the message
                    status_msg.content = f"{current_content}\n\n**Suggested Follow-up Questions:**"
                    
                    # Create actions for each question
                    actions = []
                    for question in followup_questions:
                        actions.append(
                            cl.Action(
                                name="ask_followup",
                                label=f"+ {question}",
                                payload={"question": question}
                            )
                        )
                    status_msg.actions = actions
                    await status_msg.update()
                
                # Store response in database
                db = SessionLocal()
                try:
                    chat_message = ChatMessage(
                        user_email=user_id,
                        content=status_msg.content,
                        role="assistant",
                        conversation_id=thread_id,
                        message_metadata={
                            "source": "langgraph",
                            "timestamp": time.time(),
                            "type": "rag",
                            "token_usage": result.get("rag_token_usage")
                        }
                    )
                    db.add(chat_message)
                    db.commit()
                finally:
                    db.close()

        except Exception as e:
            print(f"Error during graph execution: {str(e)}")
            status_msg.content = "An error occurred while processing your request. Please try again."
            await status_msg.update()
            
            # Store error in database
            db = SessionLocal()
            try:
                chat_message = ChatMessage(
                    user_email=user_id,
                    content=status_msg.content,
                    role="assistant",
                    conversation_id=thread_id,
                    message_metadata={
                        "source": "langgraph",
                        "timestamp": time.time(),
                        "type": "error"
                    }
                )
                db.add(chat_message)
                db.commit()
            finally:
                db.close()
            raise

    except Exception as e:
        print(f"Unexpected error in on_message: {str(e)}")
        import traceback
        traceback.print_exc()
        status_msg.content = "I apologize, but I encountered an error processing your request. Could you please try again?"
        await status_msg.update()
        
        # Store error in database
        db = SessionLocal()
        try:
            chat_message = ChatMessage(
                user_email=user_id,
                content=status_msg.content,
                role="assistant",
                conversation_id=thread_id,
                message_metadata={
                    "source": "langgraph",
                    "timestamp": time.time(),
                    "type": "error"
                }
            )
            db.add(chat_message)
            db.commit()
        finally:
            db.close()

    log_time("Total on_message", start_total)

    # Print token usage summary to terminal only
    token_msgs = []
    for key, service in [
        ("intent_token_usage", "Intent Analysis"),
        ("followup_token_usage", "Follow-up Questions"),
        ("missing_info_token_usage", "Missing Info Response"),
        ("rag_token_usage", "Knowledge Base Search")
    ]:
        usage = cl.user_session.get(key)
        if usage:
            token_msgs.append(f"Service: {service} | Model: {usage['model']} | Input tokens: {usage['input_tokens']} | Output tokens: {usage['output_tokens']}")
    if token_msgs:
        print("\n=== Token Usage Summary ===")
        for msg in token_msgs:
            print(msg)
        print("=========================")


@cl.on_chat_resume
async def on_chat_resume(thread: Dict[str, Any]):
    """
    Called when a user resumes a previous chat thread.
    Uses Chainlit's built-in data layer to retrieve chat history.
    """
    print("--- Resuming chat thread ---")
    
    try:
        # Get the thread ID from the thread data
        thread_id = thread.get("id")
        if not thread_id:
            print("Error: No thread ID found in resume data")
            await cl.Message(content="Error: Could not resume chat session. Please start a new chat.").send()
            return
        
        # Get user info
        user = cl.user_session.get("user")
        if not user:
            print("Error: No user found in session during resume")
            await cl.Message(content="Error: User not authenticated. Please log in again.").send()
            return
            
        # Fetch messages from database
        db = SessionLocal()
        try:
            messages = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.conversation_id == thread_id,
                    ChatMessage.user_email == user.identifier
                )
                .order_by(ChatMessage.created_at)
                .all()
            )
            
            chat_history = []
            for msg in messages:
                if msg.role == "user":
                    chat_history.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    chat_history.append(AIMessage(content=msg.content))
        finally:
            db.close()
        
        # Store the rebuilt chat history in session
        cl.user_session.set("chat_history", chat_history)
        
        print(f"Resumed chat with {len(chat_history)} messages.")
        
        # Store the graph in the session
        cl.user_session.set("graph", chatbot_graph)
        cl.user_session.set("thread_id", thread_id)
        
        # Send a resume message
        resume_message = "Chat session resumed. How can I help you with DSA today?"
        await cl.Message(content=resume_message).send()
        
    except Exception as e:
        print(f"Error in on_chat_resume: {e}")
        import traceback
        traceback.print_exc()
        await cl.Message(content="Error: Could not resume chat session. Please start a new chat.").send()

@cl.action_callback("ask_followup")
async def on_followup_action(action):
    """Handle follow-up question clicks."""
    # Get the question from the action payload
    question = action.payload.get("question")
    if not question:
        return
    
    # Get user from session instead of message author
    user = cl.user_session.get("user")
    if not user:
        await cl.Message(content="Error: User not authenticated. Please log in again.").send()
        return
    
    user_email = user.identifier  # This is the email from OAuth
    
    # Create a new message with the follow-up question
    followup_msg = cl.Message(content=question)
    await followup_msg.send()

    # Ensure user exists in database and check prompt limits
    db_user = ensure_user_in_db(user_email)
    can_proceed, limit_message = update_prompt_count(user_email)
    
    if not can_proceed:
        await followup_msg.update(content=limit_message)
        return

    # Show the limit message to the user
    await cl.Message(content=limit_message).send()

    # Get or create chat history
    chat_history = cl.user_session.get("chat_history", [])
    
    print(f"\n\n===== PROCESSING FOLLOW-UP QUESTION: {question} =====\n")
    start_total = time.time()

    # Get the graph from the user's session
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
        
    if not graph or not thread_id:
        await cl.Message(content="Error: Session not properly initialized. Please refresh the page.").send()
        return
        
    # Store the follow-up question in the database
    db = SessionLocal()
    try:
        # Store the message
        chat_message = ChatMessage(
            user_email=user_email,  # Use the email from session
            content=question,
            role="user",
            conversation_id=thread_id,
            message_metadata={
                "source": "langgraph",
                "timestamp": time.time(),
                "type": "followup"
            }
        )
        db.add(chat_message)
        db.commit()
    finally:
        db.close()

    # Configure the graph execution
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_email  # Use the email from session
        }
    }

    # Create initial state with chat history from database
    state = UserState(
        message=question,
        chat_history=chat_history,
        status="pending"
    )

    try:
        # Create a single message for streaming updates
        status_msg = cl.Message(content="")
        await status_msg.send()

        # Update status for intent analysis
        status_msg.content = "Analyzing your follow-up question..."
        await status_msg.update()

        try:
            # Run the graph workflow
            result = await chatbot_graph.ainvoke(state)
            
            # Handle the result based on the path taken
            if result.get("error"):
                status_msg.content = f"I apologize, but I encountered an error: {result['error']}"
                await status_msg.update()
                
                # Store error in database
                db = SessionLocal()
                try:
                    chat_message = ChatMessage(
                        user_email=user_email,  # Use the email from session
                        content=status_msg.content,
                        role="assistant",
                        conversation_id=thread_id,
                        message_metadata={
                            "source": "langgraph",
                            "timestamp": time.time(),
                            "type": "error"
                        }
                    )
                    db.add(chat_message)
                    db.commit()
                finally:
                    db.close()
                return

            # Send RAG response
            main_response = result.get("response", "I apologize, but I couldn't generate a response at this time.")
            lines = main_response.split('\n')
            current_content = ""
            
            for line in lines:
                current_content += line + "\n"
                status_msg.content = current_content
                await status_msg.update()
                await asyncio.sleep(0.1)  # Small delay between lines
            
            # Generate follow-up questions only if they're not already in the response
            followup_questions = generate_rag_followup_questions(main_response, state.chat_history)
            
            # Add follow-up questions as clickable actions
            if followup_questions:
                # Add the header as part of the message
                status_msg.content = f"{current_content}\n\n**Suggested Follow-up Questions:**"
                
                # Create actions for each question
                actions = []
                for question in followup_questions:
                    actions.append(
                        cl.Action(
                            name="ask_followup",
                            label=f"+ {question}",
                            payload={"question": question}
                        )
                    )
                status_msg.actions = actions
                await status_msg.update()
            
            # Store response in database
            db = SessionLocal()
            try:
                chat_message = ChatMessage(
                    user_email=user_email,  # Use the email from session
                    content=status_msg.content,
                    role="assistant",
                    conversation_id=thread_id,
                    message_metadata={
                        "source": "langgraph",
                        "timestamp": time.time(),
                        "type": "followup_response",
                        "token_usage": result.get("rag_token_usage")
                    }
                )
                db.add(chat_message)
                db.commit()
            finally:
                db.close()

        except Exception as e:
            print(f"Error during graph execution: {str(e)}")
            status_msg.content = "An error occurred while processing your follow-up question. Please try again."
            await status_msg.update()
            
            # Store error in database
            db = SessionLocal()
            try:
                chat_message = ChatMessage(
                    user_email=user_email,  # Use the email from session
                    content=status_msg.content,
                    role="assistant",
                    conversation_id=thread_id,
                    message_metadata={
                        "source": "langgraph",
                        "timestamp": time.time(),
                        "type": "error"
                    }
                )
                db.add(chat_message)
                db.commit()
            finally:
                db.close()
            raise

    except Exception as e:
        print(f"Unexpected error in on_followup_action: {str(e)}")
        import traceback
        traceback.print_exc()
        status_msg.content = "I apologize, but I encountered an error processing your follow-up question. Could you please try again?"
        await status_msg.update()
        
        # Store error in database
        db = SessionLocal()
        try:
            chat_message = ChatMessage(
                user_email=user_email,  # Use the email from session
                content=status_msg.content,
                role="assistant",
                conversation_id=thread_id,
                message_metadata={
                    "source": "langgraph",
                    "timestamp": time.time(),
                    "type": "error"
                }
            )
            db.add(chat_message)
            db.commit()
        finally:
            db.close()

    log_time("Total follow-up processing", start_total)

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    pass

def update_prompt_count(email: str) -> Tuple[bool, str]:
    """
    Update the prompt count for a user and check if they've exceeded their limit.
    Returns (can_proceed, message)
    """
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return True, "User not found, proceeding without limit"
            
        # Set appropriate limit based on user type
        if user.is_test_user:
            effective_limit = TEST_USER_PROMPT_LIMIT
        elif user.is_premium:
            effective_limit = user.prompt_limit  # Use stored limit for premium users
        else:
            effective_limit = DEFAULT_PROMPT_LIMIT  # Normal users get default prompts
            
        if user.prompt_count >= effective_limit:
            if user.is_test_user:
                return False, f"You've reached your test user limit of {effective_limit} prompts."
            elif user.is_premium:
                return False, f"You've reached your premium limit of {effective_limit} prompts."
            else:
                return False, f"You've reached your limit of {effective_limit} prompts. Please upgrade to premium for more."
                
        user.prompt_count += 1
        user.last_prompt_time = func.now()
        db.commit()
        
        prompts_left = effective_limit - user.prompt_count
        user_type = "test user" if user.is_test_user else "premium user" if user.is_premium else "free user"
        return True, f"You have {prompts_left} prompts remaining ({user_type})."
    except Exception as e:
        print(f"Error updating prompt count: {str(e)}")
        return True, "Error tracking prompts, proceeding without limit"
    finally:
        db.close()

def ensure_user_in_db(email: str) -> User:
    """Ensure user exists in database and return user object"""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                prompt_count=0,
                is_active=True,
                is_premium=False,
                is_test_user=False,
                prompt_limit=DEFAULT_PROMPT_LIMIT
            )
            db.add(user)
            db.commit()
        return user
    finally:
        db.close()