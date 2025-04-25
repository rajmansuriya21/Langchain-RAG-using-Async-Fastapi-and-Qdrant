# General Imports
import os
import time
from dotenv import load_dotenv
from uuid import uuid4 
import asyncio

# Langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Qdrant imports
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Cache imports
from utils.cache_utils import cached_embedding, cached_retrieval

from services.logger import logger
from uuid import uuid4

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
qdrant_db_path=os.getenv("qdrant_db_path")

class DocumentIndexer:
    def __init__(self, qdrant_db_path):
        self.db_path = qdrant_db_path
        # Store both raw and cached embeddings
        self._raw_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
        self.embedding_function = CachedEmbeddings(self._raw_embeddings)
        self.vectors = None
        self.client = AsyncQdrantClient(self.db_path)
        self.collection_name = "rag_demo_collection"

    async def index_in_qdrantdb(self, extracted_text, file_name, doc_type, chunk_size):
        """
        Index extracted text in Qdrant with optimized chunking and metadata handling.
        """
        try:
            # Create document with metadata
            doc = Document(
                page_content=extracted_text,
                metadata={"file_name": file_name, "doc_type": doc_type}
            )

            # Improved chunking strategy with optimal overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=int(chunk_size * 0.1),  # 10% overlap
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                add_start_index=True,
            )
            
            chunks = text_splitter.split_documents([doc])
            
            # Add chunk position metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "start_index": chunk.metadata.get("start_index", 0),
                    "timestamp": time.time()
                })

            # Generate UUIDs for all chunks
            uuids = [f"{str(uuid4())}" for _ in range(len(chunks))]

            # Ensure collection exists with proper configuration
            collections = await self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
                )

            # Use raw embeddings for vector store initialization
            self.vectors = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=self._raw_embeddings,  # Use raw embeddings here
                url=self.db_path
            )

            # Batch index documents
            await self.vectors.aadd_documents(documents=chunks, ids=uuids)
            
            logger.info(f"Successfully indexed {len(chunks)} chunks in QdrantDB")
            return True

        except Exception as e:
            logger.error(f"Error indexing document in QdrantDB: {e}")
            raise

    @cached_retrieval()
    async def get_retriever(self, top_k):
        """Get a cached retriever for querying the indexed documents with MMR search."""
        try:
            if self.vectors is None:
                self.vectors = QdrantVectorStore.from_existing_collection(
                    collection_name=self.collection_name,
                    embedding=self._raw_embeddings,  # Use raw embeddings here too
                    url=self.db_path
                )
            
            return self.vectors.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": top_k * 2,
                    "lambda_mult": 0.7
                }
            )
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise

class CachedEmbeddings:
    """Wrapper class for caching embeddings"""
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
    
    @cached_embedding()
    async def aembed_documents(self, texts):
        return await self.embedding_function.aembed_documents(texts)
    
    @cached_embedding()
    async def aembed_query(self, text):
        return await self.embedding_function.aembed_query(text)
