from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from sqlalchemy.engine.url import URL
from langchain.schema import HumanMessage, AIMessage
from database.models import User
from dotenv import load_dotenv

# Configure database URL
load_dotenv()

DATABASE_URL_BASE = "DATABASE_URL_BASE"

# Create engine
engine = create_engine(DATABASE_URL_BASE)

def ensure_user_in_db(email: str) -> User:
    """
    Checks if a user exists in the custom 'user' table and creates them if not.
    """
    with Session(engine) as session:
        user = session.query(User).filter(User.email == email).first()
        if not user:
            print(f"User '{email}' not found in custom DB. Creating new record.")
            new_user = User(
                email=email,
                prompt_count=0,
                prompt_limit=50,  # Default limit
                is_premium=False,
                is_test_user=False,
                is_active=True,
                last_login=func.now()
            )
            session.add(new_user)
            session.commit()
            return new_user
        else:
            # Update last_login time
            user.last_login = func.now()
            session.commit()
            return user

def update_prompt_count(email: str) -> tuple[bool, str]:
    """
    Update the prompt count for a user and check if they've exceeded their limit.
    Returns (can_proceed, message)
    """
    try:
        with Session(engine) as session:
            user = session.query(User).filter(User.email == email).first()
            if not user:
                return True, "User not found, proceeding without limit"
            
            # Set appropriate limit based on user type
            if user.is_test_user:
                effective_limit = 5000  # Test user limit
            elif user.is_premium:
                effective_limit = user.prompt_limit  # Use stored limit for premium users
            else:
                effective_limit = 50  # Normal users get default prompts
            
            if user.prompt_count >= effective_limit:
                if user.is_test_user:
                    return False, f"You've reached your test user limit of {effective_limit} prompts."
                elif user.is_premium:
                    return False, f"You've reached your premium limit of {effective_limit} prompts."
                else:
                    return False, f"You've reached your limit of {effective_limit} prompts. Please upgrade to premium for more."
            
            user.prompt_count += 1
            user.last_prompt_time = func.now()
            session.commit()
            
            prompts_left = effective_limit - user.prompt_count
            user_type = "test user" if user.is_test_user else "premium user" if user.is_premium else "free user"
            return True, f"You have {prompts_left} prompts remaining ({user_type})."
    except Exception as e:
        print(f"Error updating prompt count: {str(e)}")
        return True, "Error tracking prompts, proceeding without limit"

def save_message_to_db(email: str, conversation_id: str, content: str, role: str, metadata: Optional[Dict] = None) -> None:
    """
    Save a message to the database.
    """
    try:
        with Session(engine) as session:
            # Get user
            user = session.query(User).filter(User.email == email).first()
            if not user:
                print(f"Warning: User {email} not found when saving message")
                return
            
            # Create message record
            from database.models import Message
            message = Message(
                user_id=user.id,
                conversation_id=conversation_id,
                content=content,
                role=role,
                metadata=metadata,
                created_at=func.now()
            )
            session.add(message)
            session.commit()
    except Exception as e:
        print(f"Error saving message to database: {str(e)}")

def load_chat_history(email: str, conversation_id: str) -> List[Union[HumanMessage, AIMessage]]:
    """
    Load chat history from database.
    """
    try:
        with Session(engine) as session:
            from database.models import Message
            messages = (
                session.query(Message)
                .join(User)
                .filter(User.email == email)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )
            
            chat_history = []
            for msg in messages:
                if msg.role == "user":
                    chat_history.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    chat_history.append(AIMessage(content=msg.content))
            return chat_history
    except Exception as e:
        print(f"Error loading chat history: {str(e)}")
        return [] 