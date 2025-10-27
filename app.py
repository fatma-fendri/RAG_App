import os
import shutil
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import all necessary components from rag_core
from rag_core import (
    AdvancedRAGPipeline,
    ChromaDBManager,
    LanguageModel,
    process_all_pdfs,
    split_documents,
    load_and_store_documents 
)

# --- Global RAG Variables ---
adv_rag = None
llm_wrapper = None
rag_retriever = None
# ----------------------------

# --- FastAPI Setup ---
app = FastAPI(
    title="RAG Chatbot (Groq & ChromaDB)",
    description="A Retrieval-Augmented Generation service for documents.",
    version="1.0.0",
)

# Add CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Setup (Frontend) ---
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

DOC_DIR = "rag_documents"
TEMP_DIR = "temp_rag_docs" 

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serves the main HTML page for the chatbot interface."""
    return FileResponse("frontend/index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    """Serves the favicon."""
    return FileResponse("frontend/favicon.ico")

# --- Startup Event: Initialize RAG Pipeline ---

@app.on_event("startup")
def startup_event():
    """Initializes the LLM and RAG pipeline on application startup."""
    global adv_rag, llm_wrapper, rag_retriever 
    
    print("\n" + "="*50)
    print("--- API Startup: Initializing RAG Pipeline ---")
    print("="*50)

    try:
        # 1. Initialize ChromaDB Manager (Retriever)
        rag_retriever = ChromaDBManager(
            path="./chroma_db", 
            collection_name="rag_document_store"
        )
        
        # 2. Load Documents from initial directory
        initial_documents = process_all_pdfs(DOC_DIR)
        
        if initial_documents:
            chunks = split_documents(initial_documents)
            rag_retriever.add_documents(chunks) 
        else:
             print("No initial documents found in the document directory. RAG context is empty.")
        
        # 3. Initialize LLM (Groq)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        
        llm_wrapper = LanguageModel(groq_api_key=groq_api_key)
        
        # 4. Initialize Advanced RAG Pipeline
        adv_rag = AdvancedRAGPipeline(
            rag_retriever=rag_retriever.as_retriever(), 
            llm_wrapper=llm_wrapper
        )
        
        print("\n" + "="*50)
        print("--- API Startup Complete: RAG Pipeline READY ---")
        print("="*50)
        
    except Exception as e:
        print(f"\nFATAL RAG INITIALIZATION ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG system failed to initialize: {e}"
        )


# --- Endpoint for Document Upload ---
@app.post("/upload-pdf", tags=["RAG Admin"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, processes it, and replaces the entire RAG 
    pipeline's vector store context with the new document.
    """
    global adv_rag, rag_retriever, llm_wrapper 

    if rag_retriever is None or llm_wrapper is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized or failed to start."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are accepted."
        )

    # 1. Save the uploaded file temporarily
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Update the RAG Pipeline by calling the orchestrator function
        print(f"\n--- Updating RAG pipeline with new document: {file.filename} ---")
        
        # CRITICAL LINE: Expects two values from the function
        total_chunks, new_collection = load_and_store_documents(
            paths_to_process=[file_path], 
            db_manager=rag_retriever
        )

        # 3. CRITICAL FIX: REBUILD the RAG pipeline with the new collection reference
        
        # A. Update the ChromaDBManager's collection attribute to the new one
        rag_retriever.collection = new_collection
        
        # B. Re-create the AdvancedRAGPipeline instance
        adv_rag = AdvancedRAGPipeline(
            rag_retriever=rag_retriever.as_retriever(), 
            llm_wrapper=llm_wrapper 
        )
        print("RAG Pipeline successfully re-initialized with new document context.")
        
        # ----------------------------------------------------------------------

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "RAG pipeline successfully updated with new document.",
                "filename": file.filename,
                "total_chunks": total_chunks
            }
        )
    except Exception as e:
        print(f"Error during PDF processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF file. Error: {str(e)}"
        )
    finally:
        # 4. Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")


# --- Endpoint for Chat Query ---
@app.post("/query", tags=["RAG Chatbot"], response_model=Dict[str, Any])
def run_rag_query(query_data: Dict[str, Any]):
    """
    Processes a user question using the initialized RAG pipeline.
    """
    if adv_rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized or failed to start."
        )

    question = query_data.get("question")
    top_k = query_data.get("top_k", 5)
    summarize = query_data.get("summarize", False)

    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        # Run the RAG query
        result = adv_rag.query(
            question=question,
            top_k=top_k,
            summarize=summarize
        )
        return result
    except Exception as e:
        print(f"Error during RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(DOC_DIR):
        os.makedirs(DOC_DIR)
        print(f"Created initial document directory: {DOC_DIR}")
        print(f"!!! Please place your initial PDF files inside the '{DOC_DIR}' folder !!!")
        
    # Standard Uvicorn run command
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)