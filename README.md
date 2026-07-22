🩺 MedRAG: Retrieval-Augmented Generation for Medical Question Answering
Overview

MedRAG is a Retrieval-Augmented Generation (RAG) application developed for a hackathon to provide context-aware responses to medical queries using a trusted medical knowledge base. Instead of relying solely on the language model's internal knowledge, MedRAG retrieves relevant information from a curated medical dataset and uses it as context for generating accurate and explainable answers.

Disclaimer: This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

Features
🔍 Semantic search using vector embeddings
📚 Retrieval-Augmented Generation (RAG) pipeline
🧠 Context-aware medical question answering
⚡ Fast retrieval with vector databases
💬 Conversational interface
📄 Source-aware responses based on retrieved documents
Architecture
                    User Query
                         │
                         ▼
               Query Embedding Model
                         │
                         ▼
                 Vector Database Search
                         │
         Top-k Relevant Medical Documents
                         │
                         ▼
                Context + User Question
                         │
                         ▼
                 Large Language Model
                         │
                         ▼
                 Final Generated Answer
Tech Stack
Category	Technologies
Language	Python
Framework	Flask / Streamlit
LLM	Hugging Face Transformers
Embeddings	Sentence Transformers
Vector Store	FAISS
Data Processing	Pandas, NumPy
Backend	Python
Frontend	HTML, CSS, JavaScript
Dataset

The project uses a curated medical knowledge dataset containing disease descriptions, symptoms, precautions, medications, and treatment-related information.

Example fields include:

Disease Name
Symptoms
Description
Precautions
Medications
Recommended Diet
Suggested Workouts

The dataset is embedded into a vector database, enabling semantic retrieval based on user queries.

Workflow
Load and preprocess the medical dataset.
Generate embeddings for each medical document.
Store embeddings in a FAISS vector index.
Convert user questions into embeddings.
Retrieve the most relevant documents.
Combine retrieved context with the user's question.
Generate a context-aware response using the language model.
