from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import warnings
import logging
import zipfile
import os
import re
import shutil
from time import perf_counter
from typing import List, Dict, Any, Tuple
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ----- Optimized Configuration -----
CONFIG = {
    "zip_path": "C:/Users/nitro/Downloads/HackACure-Dataset.zip",
    "extract_dir": "data",
    "persist_dir": "db_new",
    "max_pdf_pages": 50,  # Reduced for faster processing
    "force_rebuild": True,
    
    # Using a balanced model that's better than MiniLM but more efficient than mpnet
    "embed_model": "sentence-transformers/all-MiniLM-L12-v2",  # Better than L6, more efficient than mpnet
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    
    # Optimized processing settings
    "chunk_size": 800,
    "chunk_overlap": 100,
    "max_retrieval_docs": 12,
    "max_rerank_docs": 5,
}

# ----- Enhanced Logging -----
class ChatbotLogger:
    def __init__(self):
        self.logger = logging.getLogger("MedicalChatbot")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg):
        self.logger.info(msg)
        print(f"ℹ️ {msg}", flush=True)
    
    def warning(self, msg):
        self.logger.warning(msg)
        print(f"⚠️ {msg}", flush=True)
    
    def error(self, msg):
        self.logger.error(msg)
        print(f"❌ {msg}", flush=True)
    
    def success(self, msg):
        self.logger.info(msg)
        print(f"✅ {msg}", flush=True)

logger = ChatbotLogger()

# ----- Quiet noisy logs -----
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class EfficientRAGSystem:
    """Efficient RAG system with reliable database building"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_db = None
        self.embeddings = None
        self.reranker = None
        
        self.setup_components()
    
    def setup_components(self):
        """Initialize components with reliable setup"""
        print("🎯 Setting up improved embedding model...")
        
        # Use a model that balances quality and performance
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embed_model"],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Smaller batches for stability
            }
        )
        
        # Reranker
        try:
            self.reranker = CrossEncoder(self.config["reranker_model"])
            logger.success("Reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker not available: {e}")
        
        # Build or load database
        needs_rebuild = (
            self.config["force_rebuild"] or 
            not os.path.exists(self.config["persist_dir"]) or
            not any(Path(self.config["persist_dir"]).iterdir())
        )
        
        if needs_rebuild:
            print("🔄 Building optimized knowledge base...")
            self.build_optimized_database()
        else:
            self.load_vector_store()
    
    def build_optimized_database(self):
        """Build database with optimized settings"""
        # Clean up existing database
        if os.path.exists(self.config["persist_dir"]):
            shutil.rmtree(self.config["persist_dir"])
        
        # Extract files if needed
        if not os.path.exists(self.config["extract_dir"]) or not os.listdir(self.config["extract_dir"]):
            if os.path.exists(self.config["zip_path"]):
                print("📦 Extracting dataset...")
                with zipfile.ZipFile(self.config["zip_path"], 'r') as zip_ref:
                    zip_ref.extractall(self.config["extract_dir"])
        
        # Find PDF files
        pdf_files = []
        for root, _, files in os.walk(self.config["extract_dir"]):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        # Process in smaller batches
        all_chunks = []
        total_processed = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"📖 Processing {i}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
            
            documents = self.load_pdf_document(pdf_file)
            if documents:
                # Create chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config["chunk_size"],
                    chunk_overlap=self.config["chunk_overlap"]
                )
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
                total_processed += len(chunks)
                print(f"   Created {len(chunks)} chunks (Total: {total_processed})")
        
        if not all_chunks:
            raise Exception("No content extracted from PDFs")
        
        print(f"🎯 Total chunks for vectorization: {len(all_chunks)}")
        
        # Build vector store in smaller batches if needed
        print("🗄️ Creating vector database (this may take a few minutes)...")
        
        try:
            # First attempt with all chunks
            self.vector_db = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=self.config["persist_dir"]
            )
            print("✅ Vector database created successfully!")
            
        except Exception as e:
            print(f"⚠️ First attempt failed: {e}")
            print("🔄 Trying with smaller batch...")
            
            # Try with first 1000 chunks
            try:
                self.vector_db = Chroma.from_documents(
                    documents=all_chunks[:1000],
                    embedding=self.embeddings,
                    persist_directory=self.config["persist_dir"]
                )
                print("✅ Vector database created with first 1000 chunks!")
            except Exception as e2:
                print(f"❌ Failed to create vector database: {e2}")
                raise
    
    def load_pdf_document(self, file_path: str) -> List[Document]:
        """Load a single PDF document"""
        if PdfReader is None:
            return []
        
        documents = []
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            pages_to_process = min(self.config["max_pdf_pages"], total_pages)
            
            for page_num in range(pages_to_process):
                try:
                    text = reader.pages[page_num].extract_text()
                    if text and len(text.strip()) > 100:
                        # Clean text
                        text = re.sub(r'\s+', ' ', text)
                        text = re.sub(r'-\n', '', text)
                        text = text.strip()
                        
                        documents.append(Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "file_name": os.path.basename(file_path)
                            }
                        ))
                except Exception:
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not process {os.path.basename(file_path)}: {e}")
        
        return documents
    
    def load_vector_store(self):
        """Load existing vector store"""
        self.vector_db = Chroma(
            persist_directory=self.config["persist_dir"],
            embedding_function=self.embeddings
        )
        logger.success("Vector store loaded")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents"""
        try:
            docs = self.vector_db.similarity_search(
                query, 
                k=self.config["max_retrieval_docs"]
            )
            
            if self.reranker and len(docs) > 1:
                pairs = [[query, doc.page_content] for doc in docs]
                scores = self.reranker.predict(pairs)
                ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                docs = [doc for doc, _ in ranked_docs[:self.config["max_rerank_docs"]]]
            else:
                docs = docs[:self.config["max_rerank_docs"]]
            
            return docs
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def extract_quality_answer(self, query: str, docs: List[Document]) -> str:
        """Extract quality answer from documents"""
        if not docs:
            return None
        
        # Find the most relevant content
        best_content = []
        query_terms = set(re.findall(r'\w{3,}', query.lower())) - {
            'what', 'is', 'are', 'the', 'this', 'that'
        }
        
        for doc in docs:
            content = doc.page_content
            
            # Split into meaningful sections
            paragraphs = re.split(r'\n\s*\n', content)
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if len(paragraph) > 100 and len(paragraph) < 800:
                    # Check relevance
                    paragraph_terms = set(re.findall(r'\w{3,}', paragraph.lower()))
                    overlap = len(query_terms & paragraph_terms)
                    
                    if overlap >= 2:  # Good overlap
                        best_content.append({
                            'content': paragraph,
                            'overlap': overlap,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'page': doc.metadata.get('page', 'N/A')
                        })
        
        if best_content:
            # Sort by relevance
            best_content.sort(key=lambda x: x['overlap'], reverse=True)
            
            # Use the best content
            answer = best_content[0]['content']
            
            # Format sources
            sources = []
            seen_sources = set()
            for item in best_content[:3]:
                source_key = f"{item['source']}_{item['page']}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    source_name = os.path.basename(item['source'])
                    sources.append(f"• {source_name} (Page {item['page']})")
            
            return f"{answer}\n\n📚 Sources:\n" + "\n".join(sources)
        
        return None
    
    def query(self, question: str) -> str:
        """Main query function"""
        start_time = perf_counter()
        
        logger.info(f"Processing: {question}")
        
        # Retrieve documents
        retrieved_docs = self.retrieve_documents(question)
        
        if not retrieved_docs:
            return "I couldn't find relevant information about this topic. Please try rephrasing your question."
        
        # Extract quality answer
        answer = self.extract_quality_answer(question, retrieved_docs)
        
        # Fallback
        if not answer:
            best_doc = retrieved_docs[0]
            source_name = os.path.basename(best_doc.metadata.get('source', 'Unknown'))
            page = best_doc.metadata.get('page', 'N/A')
            answer = f"{best_doc.page_content[:500]}...\n\n📚 Source: {source_name} (Page {page})"
        
        processing_time = perf_counter() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return answer

class MedicalChatbot:
    """Main chatbot class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the system"""
        try:
            self.rag_system = EfficientRAGSystem(self.config)
            logger.success("Medical chatbot ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def get_response(self, user_input: str) -> str:
        """Get response for user input"""
        try:
            user_input_lower = user_input.lower().strip()
            
            if user_input_lower in ['exit', 'quit', 'bye']:
                return "Goodbye! Stay healthy! 👋"
            
            elif user_input_lower == 'reset':
                return "Chat history cleared!"
            
            elif user_input_lower == 'help':
                return self.get_help_message()
            
            elif user_input_lower == 'status':
                return "✅ System is running with improved medical knowledge base"
            
            return self.rag_system.query(user_input)
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def get_help_message(self) -> str:
        return """
🏥 Medical Knowledge Chatbot - Improved Version

I provide answers from medical literature using enhanced embedding models.

**Try asking:**
• "What is the age of inactivity and obesity?"
• "Explain cardiac excitability"
• "What causes hypertension?"
• "What are diabetes symptoms?"

**Commands:**
• 'help' - Show this message
• 'reset' - Clear chat history  
• 'status' - Check system status
• 'exit' - End conversation
"""

def main():
    """Main application"""
    print("\n" + "="*60)
    print("🏥 Medical Knowledge Chatbot - Reliable Version")
    print("="*60)
    print("Building database with optimized settings...")
    print("Using all-MiniLM-L12-v2 for better accuracy")
    print("="*60)
    
    try:
        chatbot = MedicalChatbot(CONFIG)
        
        print("\n" + "="*60)
        print("✅ System Ready! Ask medical questions.")
        print("="*60)
        print("Type 'help' for assistance\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye! 👋")
                    break
                
                response = chatbot.get_response(user_input)
                print(f"\nBot: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nBot: Error: {e}\n")
    
    except Exception as e:
        print(f"\n❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
