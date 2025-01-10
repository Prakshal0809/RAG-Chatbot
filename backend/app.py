from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
import fitz
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Replace with environment variable

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Pinecone and OpenAI chat model
chat = ChatOpenAI(model='gpt-3.5-turbo')
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Routes
@app.route('/', methods=['GET'])
def home():
    """Root endpoint for testing."""
    return jsonify({'message': 'Welcome to the RAG Chatbot API!'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Log to check if file is saved correctly
    print(f"File saved successfully at {filepath}")

    # Process PDF and store in Pinecone
    try:
        text = process_pdf(filepath)
        print("PDF text extracted successfully.")
        store_in_pinecone(text, filename)
        print(f"Stored in Pinecone for file {filename}.")
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': f'Error processing file: {e}'}), 500
    
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Chat endpoint for handling user queries."""
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    augmented_query = augment_prompt(query)
    response = generate_response(augmented_query)
    
    if not response:
        return jsonify({'error': 'Failed to generate response from OpenAI'}), 500
    
    return jsonify({
        'response': response,
        'augmentedQuery': augmented_query
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working."""
    return jsonify({'message': 'Hello from the RAG Chatbot API!'})

# Helper Functions
def process_pdf(filepath):
    """Extract text from a PDF."""
    text = ""
    doc = fitz.open(filepath)
    for page in doc:
        text += page.get_text()
    return text

def store_in_pinecone(text, filename):
    """Store processed data in Pinecone index."""
    index_name = 'rag-chatbot-index'

    # Check if the index already exists
    if index_name not in pc.list_indexes():
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,  # Adjust the embedding dimensions if needed
                metric='dotproduct',
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index '{index_name}' created.")
        except Exception as e:
            print(f"Error creating index: {e}")
            return jsonify({'error': 'Error creating Pinecone index'}), 500
    else:
        print(f"Index '{index_name}' already exists. Skipping creation.")

    try:
        # Insert the data into Pinecone
        index = pc.Index(index_name)
        embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        embeddings = embed_model.embed_documents([text])
        
        metadata = {'text': text, 'source': 'PDF Document', 'title': filename}
        doc_id = f"pdf-{filename.split('.')[0]}"
        index.upsert(vectors=[(doc_id, embeddings[0], metadata)])
        print(f"Upserted document with ID {doc_id}.")
    except Exception as e:
        print(f"Error during Pinecone upsert: {e}")
        return jsonify({'error': 'Error during Pinecone upsert'}), 500

def augment_prompt(query):
    """Augment user query with context from Pinecone."""
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    index = pc.Index('rag-chatbot-index')
    vectorstore = LangchainPinecone(index, embed_model, "text")
    
    results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in results])
    
    return f"""Using the contexts below, answer the query.

Contexts:
{source_knowledge}

Query: {query}"""

def generate_response(augmented_query):
    """Generate response using OpenAI."""
    try:
        prompt = HumanMessage(content=augmented_query)
        res = chat.invoke([SystemMessage(content="You are a helpful assistant.")] + [prompt])
        print(f"OpenAI response generated: {res.content}")
        return res.content
    except Exception as e:
        print(f"Error generating response from OpenAI: {e}")
        return None

# Main entry point
if __name__ == '__main__':
    # Flask app is ready to run on Heroku (production-ready)
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5001)))
    
