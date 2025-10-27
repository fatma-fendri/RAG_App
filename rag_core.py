import os
from typing import List, Dict, Any, Union
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable


# --- Document Processing Functions ---

def process_all_pdfs(pdf_directory: str) -> List[Document]:
    """
    Process all PDF files in a directory using PyPDFLoader.
    (Used for initial startup loading from a directory)
    """
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
                doc.metadata['source'] = str(pdf_file)

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks


# --- Manager Classes ---

class EmbeddingManager:
    """Manages the SentenceTransformer model and embedding generation."""
    def __init__(self, model_name: str = "all-miniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_sentence_embedding_dimension()
            print(f"Model Loaded successfully. Embedding dimension: {dim}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate Embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Model not Loaded")

        print(f"Generating embeddings for {len(texts)} texts ...")
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()


class ChromaDBManager:
    """Manages the ChromaDB connection, document storage, and retrieval."""
    def __init__(self, path: str = "./chroma_db", collection_name: str = "rag_collection"):
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.embedding_manager = EmbeddingManager() 

        # Initial call to ensure the client is ready
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"ChromaDB client initialized at {self.path}. Collection: {collection_name}")

    def add_documents(self, documents: List[Document]) -> chromadb.Collection:
        """
        Adds documents to the ChromaDB collection, clearing the existing one first.
        Returns the newly created collection object.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}_{doc.metadata.get('page', 0)}" for i, doc in enumerate(documents)]

        embeddings = self.embedding_manager.generate_embeddings(texts)
        embeddings_list = embeddings.tolist()

        print(f"Adding {len(texts)} documents to ChromaDB...")
        
        # --- CRITICAL: Delete and Recreate Collection ---
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Cleared existing collection: {self.collection_name}")
        except Exception:
            pass # Ignore error if collection didn't exist

        # Recreate the collection (essential after deleting)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        # ------------------------------------------------------------------------

        self.collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print("Documents added successfully.")
        
        # Returns the newly created collection object
        return self.collection 

    def as_retriever(self) -> Any:
        """Returns a custom retriever compatible with the LangChain Runnable interface."""
        
        class CustomChromaRetriever(Runnable):
            def __init__(self, collection, embedding_manager):
                self.collection = collection
                self.embedding_manager = embedding_manager

            def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
                print(f"Retrieving top {k} documents for query: '{query[:50]}...'")
                query_embedding = self.embedding_manager.generate_embeddings([query]).tolist()[0]
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )

                retrieved_documents = []
                for doc_content, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    retrieved_documents.append(
                        Document(page_content=doc_content, metadata=metadata)
                    )
                return retrieved_documents
            
            def invoke(self, query: str, config: Dict[str, Any] = None) -> List[Document]:
                k = config.get('k', 5) if config else 5
                return self.get_relevant_documents(query, k)

        return CustomChromaRetriever(self.collection, self.embedding_manager)


class LanguageModel:
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
        print(f"Initializing LLM: {model_name} (via Groq)")
        self.llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024
        )
        print("LLM initialized.")


class AdvancedRAGPipeline:
    """The main RAG pipeline orchestrator."""
    def __init__(self, rag_retriever: Any, llm_wrapper: LanguageModel, model_name: str = "llama-3.1-8b-instant"):
        self.retriever = rag_retriever
        self.llm = llm_wrapper.llm
        self.history = [] 
        self.model_name = model_name

        self.template = """
        You are an expert Q&A system. Use the following context to answer the question.
        If you cannot find the answer in the context, clearly state that the answer is not available in the provided documents. 
        For every fact, provide a citation (e.g., [source_file: Attention.pdf - page: 2]).
        
        CONTEXT:
        {context}
        
        QUESTION: {question}
        
        ANSWER:
        """
        
        self.prompt = PromptTemplate.from_template(self.template)

        def format_docs(docs: List[Document]) -> str:
            """Formats retrieved documents into a single string with citations."""
            formatted_text = ""
            for doc in docs:
                source_file = doc.metadata.get('source_file', 'unknown_file')
                page = doc.metadata.get('page', 'unknown_page')
                citation = f" [source_file: {source_file} - page: {page}]"
                formatted_text += doc.page_content + citation + "\n---\n"
            return formatted_text

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        self.summary_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | PromptTemplate.from_template("Summarize the following answer: {context}")
            | self.llm
            | StrOutputParser()
        )
        print("RAG Pipeline initialized.")


    def query(self, question: str, top_k: int = 5, min_score: float = 0.0, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        """Executes the RAG query."""
        print(f"\n--- Running Query (k={top_k}) ---")

        retrieved_docs = self.retriever.get_relevant_documents(question, k=top_k)
        
        context_string = self.rag_chain.invoke(question, config={'k': top_k})
        
        answer_with_citations = context_string
        sources = [f"{doc.metadata.get('source_file', 'N/A')} (Page {doc.metadata.get('page', 'N/A')})" 
                   for doc in retrieved_docs]
        
        summary = None
        if summarize:
            print("Generating summary...")
            summary = self.summary_chain.invoke(context_string)

        self.history.append({
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

# =========================================================================
# === ORCHESTRATOR FUNCTION FOR DYNAMIC PDF UPLOAD ===
# =========================================================================

def load_and_store_documents(paths_to_process: List[str], db_manager: ChromaDBManager) -> tuple[int, Any]:
    """
    Loads, splits, and stores documents from a list of file paths.
    Returns: (total_chunks: int, new_collection: chromadb.Collection)
    """
    all_documents = []
    
    print(f"Found {len(paths_to_process)} files to process for RAG context update.")

    for file_path_str in paths_to_process:
        pdf_file = Path(file_path_str)
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
                doc.metadata['source'] = str(pdf_file)

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    
    if not all_documents:
        print("Warning: No documents loaded from the provided paths.")
        # CRITICAL FIX: Ensure two items are returned even if no documents are loaded
        return 0, db_manager.collection
        
    chunks = split_documents(all_documents)
    
    # Capture the newly created collection object
    new_collection = db_manager.add_documents(chunks)
    
    # CRITICAL FIX: Ensure two items are returned when documents are successfully loaded
    return len(chunks), new_collection