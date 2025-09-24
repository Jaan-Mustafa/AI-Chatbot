from sqlalchemy import text
from database.db import engine

def create_chainlit_tables():
    """Recreate the Chainlit database schema to match what Chainlit expects, dropping tables first."""
    
    with engine.connect() as conn:
        # Drop existing Chainlit tables to recreate them properly
        print("Dropping existing Chainlit tables (if they exist)...")
        conn.execute(text("DROP TABLE IF EXISTS elements CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS feedbacks CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS steps CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS threads CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS users CASCADE;"))
        
        # Create users table exactly as Chainlit expects
        print("Creating 'users' table...")
        conn.execute(text("""
            CREATE TABLE users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                identifier TEXT NOT NULL UNIQUE,
                metadata JSONB DEFAULT '{}'::jsonb,
                "createdAt" TEXT
            )
        """))
        
        # Create threads table exactly as Chainlit expects
        print("Creating 'threads' table...")
        conn.execute(text("""
            CREATE TABLE threads (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT,
                "userId" UUID REFERENCES users(id),
                "userIdentifier" TEXT REFERENCES users(identifier),
                metadata JSONB DEFAULT '{}'::jsonb,
                tags TEXT[],
                "createdAt" TEXT
            )
        """))
        
        # Create steps table exactly as Chainlit expects
        print("Creating 'steps' table...")
        conn.execute(text("""
            CREATE TABLE steps (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "threadId" UUID REFERENCES threads(id) ON DELETE CASCADE,
                "parentId" UUID,
                name TEXT,
                type TEXT NOT NULL,
                input TEXT,
                output TEXT,
                "isError" BOOLEAN DEFAULT FALSE,
                streaming BOOLEAN DEFAULT FALSE,
                "waitForAnswer" BOOLEAN DEFAULT FALSE,
                "showInput" TEXT,
                "defaultOpen" BOOLEAN DEFAULT FALSE,
                "createdAt" TEXT NOT NULL,
                start TEXT,
                "end" TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                generation JSONB,
                tags TEXT[],
                language TEXT
            )
        """))
        
        # Create feedbacks table with forId as UUID
        print("Creating 'feedbacks' table...")
        conn.execute(text("""
            CREATE TABLE feedbacks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "stepId" UUID REFERENCES steps(id) ON DELETE CASCADE,
                "forId" UUID,
                value INTEGER,
                comment TEXT,
                "createdAt" TEXT
            )
        """))
        
        # Create elements table exactly as Chainlit expects
        print("Creating 'elements' table...")
        conn.execute(text("""
            CREATE TABLE elements (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                "stepId" UUID REFERENCES steps(id) ON DELETE CASCADE,
                name TEXT,
                type TEXT,
                url TEXT,
                "objectKey" TEXT,
                mime TEXT,
                "threadId" UUID REFERENCES threads(id),
                size TEXT,
                page INTEGER,
                language TEXT,
                "forId" UUID,
                "createdAt" TEXT,
                "chainlitKey" TEXT,
                display TEXT,
                props JSONB DEFAULT '{}'::jsonb
            )
        """))
        
        # Commit all changes
        conn.commit()
        print("✅ Successfully CREATED Chainlit database schema with createdAt as TEXT and schema fixes!")
        
        # Check if tables were created
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"))
        tables = [row[0] for row in result]
        print(f"📋 Current tables: {tables}")

if __name__ == "__main__":
    create_chainlit_tables() 