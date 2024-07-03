#!/usr/bin/python
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
import uuid
from flask import Flask, request, jsonify, send_from_directory
from interfaces.runpod import RunpodServerlessLLM
from dotenv import load_dotenv
import os
from flask_cors import CORS
from werkzeug.utils import secure_filename
import __main__
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pkl'}
app = Flask(__name__, static_folder='public')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, origins="*")
app.secret_key = '3be69c0ad309fd7fa7ae46a9342775321900a493dfc52192bb103965645052b5'

class Document:
    def __init__(self, page_content, metadata, embedding=None):
        self.page_content = page_content
        self.metadata = metadata
        self.embedding = embedding

def load_cached_documents(cache_file):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Qdrant client
qdrant_client = QdrantClient(host=os.getenv('QDRANT_HOST'), port=6333)

embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={
    "trust_remote_code": True
})

# Function to query relevant documents
def query_documents(query, top_k=5):
    query_embedding = embedding_model.embed_query(query)
    results = qdrant_client.search(
        collection_name=os.getenv('COLLECTION'),
        query_vector=query_embedding,
        limit=top_k,
        score_threshold=0.6,
        search_params=models.SearchParams(hnsw_ef=128, exact=True)
    );

    return results

# Initialize Ollama model
# ollama_model = Ollama(model="llama3:8b")
ollama_model = RunpodServerlessLLM(
    pod_id=os.getenv('RUNPOD_ID'),
    api_key=os.getenv('RUNPOD_SECRET'),
)

# Function to generate an answer using Ollama
def generate_answer(query, context_docs):
    context = " ".join([doc.payload["page_content"] for doc in context_docs])
    full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nProvide the source at the end of the every answer.\n\nAnswer:"
    response = ollama_model.generate(prompts=[full_prompt])
    return response.generations[0][0].text

# Function to handle the entire process: query the documents and generate an answer
def answer_query(query):
    # Retrieve relevant documents
    search_results = query_documents(query)
    # Generate an answer based on the retrieved documents
    answer = generate_answer(query, search_results)
    return answer

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('public', path)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get('query', '')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    answer = answer_query(user_input)
    return jsonify({"response": answer})

@app.route('/lookup', methods=['POST'])
def lookup():
    data = request.json
    user_input = data.get('query', '')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    search_results = query_documents(user_input)
    data = []
    for doc in search_results:
        data.append(doc.payload["page_content"])
    return jsonify({"response": data})

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"response": 'No file'}), 401
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return jsonify({"response": 'No file'}), 401
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            docs = load_cached_documents(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            vectors = []
            payloads = []
            ids = []
            urls = []
            for doc in docs:
                vectors.append(doc.embedding)
                payloads.append({"page_content": doc.page_content, "metadata": doc.metadata})
                urls.append(doc.metadata["source"])
                ids.append(str(uuid.uuid4()))  # Generate a valid UUID for each document

            # Check if the collection exists, and create if not
            collection_name = os.getenv('COLLECTION')
            # if not qdrant_client.collection_exists(collection_name):
            max_vector = 0
            for vector in vectors:
                max_vector = max(max_vector, len(vector))

            if not qdrant_client.collection_exists(collection_name):
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=max_vector, distance=models.Distance.COSINE)
                )

            # Delete from qdrant if metadata.source matches the list of urls
            if urls:
                for url in urls:
                    filter_ = Filter(
                        must=[
                            FieldCondition(
                                key="metadata.source",
                                match=MatchValue(value=url)
                            )
                        ]
                    )
                    qdrant_client.delete(
                        collection_name=collection_name,
                        points_selector=filter_
                    )

            # Upload vectors to Qdrant
            qdrant_client.upload_collection(
                collection_name=collection_name,
                vectors=vectors,
                payload=payloads,
                ids=ids
            )

            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return jsonify({"response": filename}), 201

if __name__ == '__main__':
    app.run(debug=os.getenv('DEBUG'), port=5050, host="0.0.0.0")
__main__.Document = Document