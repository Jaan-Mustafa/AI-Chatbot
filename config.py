import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Model configuration for LLMs
INTENT_MODEL = "gpt-4o"  # Accurate, for main answers
RAG_MODEL = "gpt-4o"     # Accurate, for main answers
FOLLOWUP_MODEL = "gpt-4o-mini"  # Fast, good for follow-up

# API configuration

USER_ID = os.getenv("USER_ID")
PASSWORD = os.getenv("PASSWORD")
SALES_CHANNEL_USER_ID = os.getenv("SALES_CHANNEL_USER_ID")

# Database Configuration
# Configure Chainlit's data layer for chat history persistence
DATABASE_URL_BASE = os.getenv("DATABASE_URL_BASE")

# Initialize LLM models
main_model = ChatOpenAI(model_name=RAG_MODEL, temperature=0.7, streaming=True)
extraction_model = ChatOpenAI(model_name=INTENT_MODEL, temperature=0.1)
followup_model = ChatOpenAI(model_name=FOLLOWUP_MODEL, temperature=0.3)

# Chainlit OAuth configuration
os.environ['CHAINLIT_GOOGLE_CLIENT_ID'] = os.getenv('OAUTH_GOOGLE_CLIENT_ID')
os.environ['CHAINLIT_GOOGLE_CLIENT_SECRET'] = os.getenv('OAUTH_GOOGLE_CLIENT_SECRET')
os.environ['CHAINLIT_AUTH_SECRET'] = os.getenv('CHAINLIT_AUTH_SECRET')

# Document processing configuration
chunk_size = 1500
chunk_overlap = 150
PDF_STORAGE_PATH = "./pdfs"

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings()

# Print OAuth configuration (for debugging)
def print_oauth_config():
    print("Google Client ID:", os.getenv("OAUTH_GOOGLE_CLIENT_ID"))
    print("Google Client Secret:", os.getenv("OAUTH_GOOGLE_CLIENT_SECRET"))
    print("Chainlit Auth Secret:", os.getenv("CHAINLIT_AUTH_SECRET"))