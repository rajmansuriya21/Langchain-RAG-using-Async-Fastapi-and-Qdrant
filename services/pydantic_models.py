from fastapi import Form, UploadFile, File
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
import os
import re
load_dotenv(override=True)


class Source(BaseModel):
    file_name: str
    chunk_index: int
    content: str
    relevance_score: Optional[float] = None


class ChatRequest(BaseModel):
    username: str
    query: str
    session_id: Optional[str] = None
    no_of_chunks: Optional[int] = 3

    @field_validator('no_of_chunks')
    def validate_chunks(cls, v):
        if v is not None and (v < 1 or v > 10):
            raise ValueError('no_of_chunks must be between 1 and 10')
        return v


class ChatResponse(BaseModel):
    username: str
    query: str
    refine_query: str
    response: str
    session_id: str
    sources: List[Source]
    confidence_score: Optional[float] = None
    processing_time: float
    token_usage: Dict[str, int]
    debug_info: Optional[dict] = None

    @model_validator(mode='after')
    def validate_response(self):
        if len(self.sources) == 0 and self.response != "I don't have enough information to answer this question based on the provided context":
            raise ValueError('Response must either have sources or indicate insufficient information')
        return self