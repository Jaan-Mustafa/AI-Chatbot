from datetime import time
import time
from typing import Optional, List, Union, Dict, Any
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.prompts import ChatPromptTemplate
import asyncio
import chainlit as cl
from pathlib import Path
import re
import json

from config import (
    main_model,
    PDF_STORAGE_PATH
)

# Timing log utility
def log_time(label, start_time):
    print(f"{label} took {time.time() - start_time:.2f} seconds")

# Format chat history for prompt
def format_chat_history_for_prompt(chat_history: List[Union[HumanMessage, AIMessage]]) -> str:
    """Formats chat history for inclusion in a prompt. Only use last 3 messages for brevity."""
    formatted_lines = []
    for msg in chat_history[-3:]:  # Use last 3 messages for brevity
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"Assistant: {msg.content}")
    return "\n".join(formatted_lines)

# Utility to count tokens for a given text and model
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a given text for a specific OpenAI model.
    """
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Update Loading Message 
async def update_loading_message(loading_msg, messages, interval=5.0):
    """
    Updates a loading message with different text at specified intervals to keep the user engaged.
    """
    try:
        for message in messages:
            print(f"DEBUG: update_loading_message - Updating message to: {message}")
            loading_msg.content = message
            await loading_msg.update()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        print("DEBUG: update_loading_message - Cancelled")
        pass
    except Exception as e:
        print(f"Error updating loading message: {str(e)}")

def process_pdfs(pdf_storage_path: str):
    start = time.time()
    pdf_directory = Path(pdf_storage_path)
    docs = []  
    
    # Better text splitting strategy for DSA documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )

    for pdf_path in pdf_directory.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        
        # Add topic info to metadata
        topic_name = pdf_path.stem.split('_')[-1]
        for doc in documents:
            doc.metadata["topic"] = topic_name
        
        docs += text_splitter.split_documents(documents)

    print(f"Processed {len(docs)} document chunks from {len(list(pdf_directory.glob('*.pdf')))} PDF files")

    # Create vector store with normalized similarity search
    doc_search = Chroma.from_documents(
        docs, 
        OpenAIEmbeddings(),
        collection_name="dsa_documents"
    )

    namespace = "chromadb/dsa_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")
    log_time("PDF Processing & Vector Indexing", start)
    return doc_search

def format_docs(docs):
    # Enhanced document formatting with topic information, only top 3 docs
    formatted_docs = []
    for doc in docs[:3]:  # Only top 3 docs
        topic = doc.metadata.get("topic", "Unknown Topic")
        page = doc.metadata.get("page", "Unknown Page")
        formatted_docs.append(f"[{topic} - Page {page}]\n{doc.page_content}")
    return "\n\n" + "-"*50 + "\n\n".join(formatted_docs) + "\n" + "-"*50 + "\n\n"

# Add diagnostic command
@cl.action_callback("run_diagnostics")
async def run_diagnostics():
    """Run diagnostics to check the system status"""
    # Create a diagnostic message
    diag_msg = cl.Message(content="Running system diagnostics...")
    await diag_msg.send()
    
    # Basic system info
    diagnostics = f"""
## System Diagnostics

**Environment:**
- PDFs loaded: {len(list(Path(PDF_STORAGE_PATH).glob('*.pdf')))}

**Loading Messages:**
- Intent analysis: ⏳ Processing your request...
- Missing info: ⏳ Analyzing your request...
- Hint generation: ⏳ Generating your DSA hints...
- RAG flow: ⏳ Searching through DSA knowledge base...

If you're experiencing issues with loading messages, try refreshing the page.
"""
    
    # Update diagnostic message
    diag_msg.content = diagnostics
    await diag_msg.update()

def generate_followup_questions(chat_history: List[Union[HumanMessage, AIMessage]], current_response: str) -> List[str]:
    """
    Generate relevant followup questions based on chat history and current response.
    Returns a list of followup questions.
    """
    # Create prompt for followup questions
    prompt = ChatPromptTemplate.from_template("""
    You are DSAGPT, an expert DSA tutor. Based on the conversation history and the current response,
    generate 3 relevant followup questions that the user might want to ask next.
    
    Conversation History:
    {chat_history}
    
    Current Response:
    {current_response}
    
    Generate 3 followup questions that:
    1. Are directly related to the current DSA topic or problem
    2. Help clarify or expand on the information provided
    3. Are specific and actionable
    4. Are phrased in a natural, conversational way
    
    Return ONLY the questions as a JSON array of strings, with no additional text.
    Example format: ["Question 1?", "Question 2?", "Question 3?"]
    """)
    
    # Format chat history
    formatted_history = format_chat_history_for_prompt(chat_history)
    
    # Generate questions using the main model
    questions_str = main_model.invoke(prompt.format(
        chat_history=formatted_history,
        current_response=current_response
    ))
    
    try:
        # Parse the response as JSON
        questions = json.loads(questions_str.content)
        # Ensure we have exactly 3 questions
        return questions[:3] if len(questions) > 3 else questions
    except Exception as e:
        print(f"Error parsing followup questions: {str(e)}")
        return []