from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import uuid
import logging
from fastapi import HTTPException, APIRouter
import logging
import uuid, time
from fastapi.responses import JSONResponse
from typing import List
from fastapi import Depends
import langsmith as ls
import nest_asyncio
import os
from datetime import *
from services.pydantic_models import ChatRequest, ChatResponse, Source
from services.logger import logger
import langsmith as ls
from langsmith.wrappers import wrap_openai
from fastapi.openapi.models import Response
from typing import AsyncGenerator
from uuid import uuid4
import asyncio
from utils.db_utils import get_past_conversation_async, add_conversation_async
from utils.langchain_utils import generate_chatbot_response, index_documents
from utils.utils import extract_text_from_file
import uvicorn
import aiomonitor
from io import BytesIO
import shutil
from utils.cache_utils import initialize_cache

nest_asyncio.apply()

app = FastAPI(title="RAG API with Async FastAPI and Qdrant",
             description="A RAG system using FastAPI, Langchain, and Qdrant with async support")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize SQLite cache
        await initialize_cache()
        logger.info("Startup tasks completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG API",
        "available_endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }

@app.post("/upload-knowledge")
async def upload_knowledge(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Create unique filename with uuid
        unique_filename = f"{file.filename}_{str(uuid4())}"
        file_path = os.path.join("uploads", unique_filename)
        
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Read file content
        file_content = await file.read()
        
        # Save uploaded file permanently
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Process the file
        file_extension = file.filename.split('.')[-1].lower()
        extracted_text = await extract_text_from_file(file_content, file_extension)
            
        logger.info(f"File uploaded: {file.filename}")
        logger.info(f"Extracted text from file: {extracted_text}")

        logger.info(f"Indexing documents in QdrantDB")
        await index_documents(username, extracted_text, file.filename, file_extension)
            
        return {
            'response': 'Indexed Documents Successfully', 
            'file_path': file_path,
            'extracted_text': extracted_text
        }
        
    except Exception as e:
        logger.error(f"Error processing indexing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while indexing documents {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest
):
    try:
        start_time = datetime.now()
        logger.info(f"Request started at {start_time}")
        logger.info(f"Received request from {request.username} for question: {request.query}")
        
        # Initialize session if needed
        if request.session_id is None:
            request.session_id = str(uuid4())
            past_messages = []
        else:
            logger.info(f"Fetching past messages")
            past_messages = await get_past_conversation_async(request.session_id)
            logger.info(f"Fetched past messages: {past_messages}")

        # Generate response
        logger.info(f"Generating chatbot response")
        response, response_time, input_tokens, output_tokens, total_tokens, final_context, refined_query, extracted_documents = await generate_chatbot_response(
            request.query, 
            past_messages, 
            request.no_of_chunks, 
            request.username
        )
        
        # Process sources
        sources = []
        for doc in extracted_documents:
            sources.append(Source(
                file_name=doc.metadata.get("file_name", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", -1),
                content=doc.page_content,
                relevance_score=doc.metadata.get("relevance_score", None)
            ))

        # Add to conversation history
        logger.info(f"Adding conversation to chat history")
        await add_conversation_async(request.session_id, request.query, response)
        logger.info(f"Added conversation to chat history")

        # Calculate confidence score based on source relevance
        confidence_score = sum(source.relevance_score or 0 for source in sources) / len(sources) if sources else None

        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            username=request.username,
            query=request.query,
            refine_query=refined_query,
            response=response,
            session_id=request.session_id,
            sources=sources,
            confidence_score=confidence_score,
            processing_time=processing_time,
            token_usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            debug_info={
                "context_used": final_context,
                "retrieval_time": response_time
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An unexpected error occurred while generating chatbot response: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)