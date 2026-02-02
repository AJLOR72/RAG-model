from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging

# Optional imports with error handling
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from sentence_transformers import CrossEncoder
    DEPS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    DEPS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

CONFIG = {
    "persist_dir": "db",
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
}

class SimpleMedicalChatbot:
    def __init__(self, config):
        self.config = config
        self.vector_db = None
        self.embeddings = None
        self.initialize()
    
    def initialize(self):
        if not DEPS_AVAILABLE:
            print("❌ Required dependencies not available")
            return
            
        print("🔄 Initializing Medical Chatbot...")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config["embed_model"],
                model_kwargs={'device': 'cpu'}
            )
            
            if os.path.exists(self.config["persist_dir"]):
                self.vector_db = Chroma(
                    persist_directory=self.config["persist_dir"],
                    embedding_function=self.embeddings
                )
                print("✅ Chatbot ready!")
            else:
                print("❌ No database found. Please process data first.")
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")

chatbot = SimpleMedicalChatbot(CONFIG)

@app.route('/')
def home():
    return "Medical Chatbot Backend is running! Use /chat endpoint for queries."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not DEPS_AVAILABLE:
            return jsonify({'response': 'Backend dependencies not installed.'})
        
        if chatbot.vector_db is None:
            return jsonify({'response': 'Database not initialized. Please process medical data first.'})
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please enter a message.'})
        
        response = chatbot.get_response(user_message)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

@app.route('/status')
def status():
    status_msg = "ready" if (chatbot.vector_db is not None) else "database not loaded"
    return jsonify({'status': status_msg})

if __name__ == '__main__':
    print("🚀 Starting Medical Chatbot Server...")
    print("📍 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)