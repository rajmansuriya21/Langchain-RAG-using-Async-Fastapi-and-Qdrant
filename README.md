# Langchain RAG using Async FastAPI and Qdrant 🚀

A modern document Question-Answering system built with Langchain, AsyncIO, FastAPI, and Qdrant vector store.

## 🌟 Features

- 📚 PDF Document Upload and Processing
- 🔍 Advanced RAG (Retrieval Augmented Generation)
- ⚡ Async API Implementation with FastAPI
- 🗃️ Qdrant Vector Store Integration
- 💾 Caching System for Improved Performance
- 📝 Conversation History Tracking
- 🔄 Automatic Query Refinement
- 📊 Debug Information and Performance Metrics

## 🛠️ Tech Stack

- FastAPI
- Langchain
- Qdrant
- AsyncIO
- Python 3.11+
- SQLite (for caching and chat logs)

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager

### 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/rajmansuriya21/Langchain-RAG-using-Async-Fastapi-and-Qdrant.git
cd Langchain-RAG-using-Async-Fastapi-and-Qdrant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

## 📚 API Endpoints

### Upload Document

```http
POST /upload-knowledge
```

### Chat Interface

```http
POST /chat
```

Request body:

```json
{
  "username": "string",
  "query": "string",
  "session_id": "string",
  "no_of_chunks": 3
}
```

## 📁 Project Structure

```
├── api.py              # FastAPI application and endpoints
├── services/          # Core services
│   ├── logger.py
│   └── pydantic_models.py
├── utils/             # Utility functions
│   ├── cache_utils.py
│   ├── db_utils.py
│   ├── langchain_utils.py
│   ├── prompts.py
│   ├── qdrant_utils.py
│   └── retry_utils.py
└── uploads/           # Document storage
```

## 🌟 Features in Detail

1. **Document Processing**

   - PDF document uploading and processing
   - Text chunk extraction and vectorization

2. **RAG Implementation**

   - Advanced retrieval using Qdrant vector store
   - Context-aware response generation
   - Query refinement for better results

3. **Performance Optimization**

   - Caching system for faster responses
   - Async implementation for better scalability
   - Configurable chunk sizes for retrieval

4. **Monitoring and Debugging**
   - Detailed response metrics
   - Token usage tracking
   - Processing time information
   - Debug context information

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Author

- Raj Mansuriya ([@rajmansuriya21](https://github.com/rajmansuriya21))

---

⭐ If you find this project helpful, please give it a star!
