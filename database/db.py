from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection string for PostgreSQL
DATABASE_URL =os.getenv("POSTGRES_URI")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base() 


from sqlalchemy import text
from database.db import engine

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connected successfully:", result.scalar())
    except Exception as e:
        print("❌ Database connection failed:", str(e))

if __name__ == "__main__":
    test_connection()
