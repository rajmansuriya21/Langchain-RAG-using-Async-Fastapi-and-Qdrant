import langsmith  as ls
from langsmith.wrappers import wrap_openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback  # Updated import
import time
import os
from utils.prompts import get_query_refiner_prompt, get_main_prompt
from qdrant_client import QdrantClient,AsyncQdrantClient
from utils.qdrant_utils import DocumentIndexer
import asyncio
from services.logger import logger
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai.embeddings import OpenAIEmbeddings
from utils.retry_utils import async_retry, simple_fallback, CircuitBreaker

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
qdrant_db_path=os.getenv("qdrant_db_path")

# Initialize circuit breakers for critical operations
embedding_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=300)  # 5 minutes timeout
llm_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=180)  # 3 minutes timeout


async def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

async def index_documents(username,extracted_text,filename,file_extension):
    try:
        indexer = DocumentIndexer(qdrant_db_path)
        start_time = time.time()
        logger.info("Searching for similar documents in ChromaDB...")

        await indexer.index_in_qdrantdb(
            extracted_text=extracted_text,
            file_name=filename,
            doc_type=file_extension,
            chunk_size=1500  
        )
        logger.info(f"Document indexing completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise RuntimeError(f"Failed to process documents: {str(e)}")


@async_retry(retries=3, delay=1, backoff=2)
async def retrieve_similar_documents(refined_query: str, num_of_chunks: int, username: str) -> str:
    """Retrieve similar documents with retry logic and circuit breaker protection."""
    try:
        indexer = DocumentIndexer(qdrant_db_path)
        start_time = time.time()
        logger.info("Searching for similar documents in Qdrant...")

        if num_of_chunks is None:
            num_of_chunks = int(os.getenv('no_of_chunks', 3))
        if not isinstance(num_of_chunks, int) or num_of_chunks <= 0:
            raise ValueError(f"Invalid number of chunks: {num_of_chunks}")

        retriever = await indexer.get_retriever(top_k=num_of_chunks)
        if not retriever:
            raise ValueError("Failed to initialize document retriever")

        extracted_documents = await retriever.ainvoke(refined_query)
        
        if not extracted_documents:
            return "", []

        # Calculate relevance scores
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
        query_embedding = await embedding_breaker.call(embedding_function.aembed_query, refined_query)
        
        for doc in extracted_documents:
            try:
                doc_embedding = await embedding_breaker.call(embedding_function.aembed_query, doc.page_content)
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0]
                doc.metadata["relevance_score"] = float(similarity)
            except Exception as e:
                logger.warning(f"Could not calculate relevance score: {e}")
                doc.metadata["relevance_score"] = 0.0

        extracted_text_data = await format_docs(extracted_documents)
        logger.info(f"Document retrieval completed in {time.time() - start_time:.2f} seconds")
        return extracted_text_data, extracted_documents

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise


@async_retry(retries=3, delay=1, backoff=2, fallback=simple_fallback)
async def invoke_chain(query, context, history, llm):
    """Handles the streamed response asynchronously with retry logic."""
    return await llm_breaker.call(async_invoke_chain, query, context, history, llm)

async def async_invoke_chain(query, context, history, llm):
    """Core chain invocation with circuit breaker protection."""
    logger.info("Initializing Chain...")
    final_chain = get_main_prompt() | llm | StrOutputParser()
    input_data = {"user_query": query, "context": context, "messages": history.messages}

    with get_openai_callback() as cb:
        final_response = await final_chain.ainvoke(input_data)

    return final_response, cb


def create_history(messages):
    history = InMemoryChatMessageHistory()
    for message in messages:
            if message["role"] == "user":
                history.add_user_message(message["content"])
            else:
                history.add_ai_message(message["content"])

    return history

def initialize_llm(model=None, temperature=None, llm_provider=None):
    """Initialize the language model."""
    if temperature is None:
        temperature = os.getenv('temperature')
    if llm_provider is None:
        llm_provider = os.getenv('llm_provider')
        model= os.getenv('model')

    if llm_provider == "openai":
        logger.info(f"Initializing OpenAI model with values {model} and {temperature}")
        llm=ChatOpenAI(api_key=OPENAI_API_KEY,temperature=temperature, model_name=model,streaming=True,stream_usage=True)
    return llm


@async_retry(retries=2, delay=1, backoff=2)
async def refine_user_query(query, messages):
    """Refines the user query asynchronously with retry logic."""
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    history = create_history(messages)
    prompt = get_query_refiner_prompt()
    refined_query_chain = prompt | llm | StrOutputParser()
    return await llm_breaker.call(refined_query_chain.ainvoke, {"query": query, "messages": history.messages})


@ls.traceable(run_type="chain", name="Chat Pipeline")
async def generate_chatbot_response(query, past_messages, no_of_chunks, username):
    """Main function to generate chatbot responses with comprehensive error handling."""
    try:
        logger.info("Refining user query")
        refined_query = await refine_user_query(query, past_messages)
        logger.info(f"Generated refined query: {refined_query}")

        try:
            extracted_text_data, extracted_documents = await retrieve_similar_documents(refined_query, int(no_of_chunks), username)
            if not extracted_documents:
                return "I don't have enough information to answer this question based on the provided context", 0, 0, 0, 0, "", refined_query, []
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise ValueError(f"Failed to retrieve relevant documents: {str(e)}")

        llm = initialize_llm()
        history = create_history(past_messages)
        logger.info("Created history for session")

        logger.info("Fetching response")
        start_time = time.time()
        try:
            final_response, cb = await invoke_chain(query, extracted_text_data, history, llm)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise ValueError(f"Failed to generate response: {str(e)}")

        response_time = time.time() - start_time

        if not final_response or final_response.strip() == "":
            raise ValueError("Generated empty response")

        return (
            final_response,
            response_time,
            cb.prompt_tokens,
            cb.completion_tokens,
            cb.total_tokens,
            extracted_text_data,
            refined_query,
            extracted_documents
        )

    except Exception as e:
        logger.error(f"Error in generate_chatbot_response: {str(e)}")
        raise