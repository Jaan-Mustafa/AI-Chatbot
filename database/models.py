from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from database.db import Base

class User(Base):
    __tablename__ = "user"
    
    # Internal identifier
    id = Column(Integer, primary_key=True, index=True)
    
    # Authentication fields (email is the primary identifier from Google OAuth)
    email = Column(String, unique=True, nullable=False)  # Will be used as username
    
    # App-specific data
    prompt_count = Column(Integer, default=0)
    prompt_limit = Column(Integer, default=3)
    is_premium = Column(Boolean, default=False)
    is_test_user = Column(Boolean, default=False)
    
    # Usage tracking
    last_prompt_time = Column(DateTime(timezone=True), nullable=True)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # User status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, ForeignKey("user.email"), nullable=False)
    content = Column(Text, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    
    # Timestamp fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    conversation_id = Column(String, nullable=False)  # To group messages in same conversation
    
    # Metadata for additional info (like tokens, model used, etc)
    message_metadata = Column(JSON, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_email": self.user_email,
            "content": self.content,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "conversation_id": self.conversation_id,
            "metadata": self.message_metadata or {}
        } 