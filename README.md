# 🤖 RAG Chatbot (Retrieval-Augmented Generation)
This project implements an advanced Question-Answering (QA) solution based on the RAG (Retrieval-Augmented Generation) architecture. It allows users to upload custom PDF documents to build an ephemeral knowledge base, ensuring all answers are accurate, context-aware, and factually grounded, minimizing Large Language Model (LLM) hallucinations.
## 1. Project Architecture and Core Skills
The application is built using a modern, asynchronous architecture that separates the user interface from the core LLM and data services.
### Key Skills Demonstrated
- RAG Engineering: Designing and implementing a robust pipeline for context retrieval and factual response generation.

- Microservices API: Building a fast, scalable, and resilient API using the FastAPI/ASGI stack.

- Vector Databases: Expertise in embeddings, semantic search, and utilizing ChromaDB for knowledge storage.

- Full Stack Development: Connecting the high-performance backend to an interactive frontend (HTML/CSS/JS).

- Performance Optimization: Leveraging asynchronous programming and low-latency technologies (Groq, Uvicorn) to minimize system latency.

### Architecture Overview
| Component | Technology | Role in the Project |
| :--- | :--- | :--- |
| **Backend API** | **FastAPI** / **Uvicorn** | Manages all requests (`POST /upload-pdf`, `POST /query`) and handles core RAG logic **asynchronously**. |
| **Orchestration** | **LangChain** | Framework used to chain together the document loading, splitting, retrieval, and generation steps. |
| **Vectorization** | **Sentence Transformers** (`all-miniLM-L6-v2`) | Transforms text into **numerical vectors (embeddings)** for semantic searching. |
| **Vector DB** | **ChromaDB** | Stores all document embeddings and metadata for **quick retrieval**. |
| **LLM / Generator** | **Groq API** (e.g., Llama-3.1) | Generates the final, context-grounded response with **extremely low latency**. |
| **Frontend** | **HTML/CSS/JavaScript** | Provides the user interface for document upload and the interactive chat experience. |

## 2. Live Demo

The application provides a two-panel interface for document ingestion and real-time querying...

### 🎥 Watch the Full Demo Video (via Google Drive)

To see the RAG pipeline in action, click the image below:
[![Watch the RAG Chatbot Demo Video](https://raw.githubusercontent.com/fatma-fendri/RAG_App/main/frontend/static/images/Demo2.png)](https://drive.google.com/file/d/1znje1_VLdN78bLTUfoKw8kBc-bhMV97J/view?usp=sharing)

## 3. RAG Pipeline Flow
The project is defined by two primary asynchronous processes managed by the FastAPI backend:
### A. Context Ingestion Flow (POST /upload-pdf)
This process updates the chatbot's knowledge base when a new PDF is uploaded:

1. File Handling: FastAPI receives the PDF file stream.

2. Loading & Chunking: The document is loaded (using PyMuPDFLoader) and divided into smaller chunks using the RecursiveCharacterTextSplitter to maintain textual context.

3. Vectorization (Embedding): Each text chunk is converted into a vector embedding.

4. Storage: The embeddings and associated metadata (source file, page number) are stored in the ChromaDB collection.
### B. Query and Generation Flow (POST /query)
This is the core question-answering process:
1. Query Vectorization: The user's question is immediately converted into a vector.
2. Retrieval: The query vector is used to perform a semantic similarity search in ChromaDB, retrieving the $k$ most relevant text chunks from the entire document corpus.
3. Prompt Construction: The retrieved chunks (the context) are inserted into a template alongside the user's question to form a detailed Prompt.
4. Generation: The prompt is sent to the high-speed LLM via the Groq API, which generates a response based only on the provided context.
5. Response: The final, sourced answer is returned to the client's frontend.

## 3. Setup and Installation
### Prerequisites
- Python 3.8+
- A virtual environment (venv).
- An API Key for Groq.










