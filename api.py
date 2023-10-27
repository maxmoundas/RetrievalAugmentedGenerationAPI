# install dependencies:
# pip install python-dotenv Flask flask_restful pdfminer.six langchain streamlit plotly openai matplotlib pandas scipy scikit-learn tiktoken faiss-cpu werkzeug

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv, set_key

from ingest import ingest_docs
from app import generate_response

app = Flask(__name__)

# Load environment variables from .env file (if it exists)
load_dotenv()

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Endpoint to collect OpenAI API key
@app.route('/collect_key', methods=['POST'])
def api_collect_api():
    # Extract OpenAI API key from the form data
    openai_api_key = request.form.get('openai_api_key')
    if not openai_api_key:
        return jsonify({"status": "failure", "message": "OpenAI API key is missing from the request body"}), 400

    # Save the OpenAI API key to .env and also set it as an environment variable for the current session
    try:
        # Find the path to the .env file. Assuming it's in the same directory as this script.
        env_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.env')
        
        # Set the key in the .env file
        set_key(env_path, 'OPENAI_API_KEY', openai_api_key)

        # Also set it as an environment variable for the current session
        os.environ['OPENAI_API_KEY'] = openai_api_key

        return jsonify({"status": "success", "message": "OpenAI API key saved to .env"}), 200
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error saving OpenAI API key to .env: {str(e)}"}), 500

# Endpoint to ingest a document
@app.route('/ingest', methods=['POST'])
def api_ingest():
    # Ensure a file was uploaded
    if 'document' not in request.files:
        return jsonify({"status": "failure", "message": "No document uploaded"}), 400

    # Get the uploaded file
    uploaded_file = request.files['document']

    # Check if the uploaded file is not empty
    if uploaded_file.filename == '':
        return jsonify({"status": "failure", "message": "No selected file"}), 400

    # Check if the uploaded file has a valid filename
    if uploaded_file and allowed_file(uploaded_file.filename):
        # Create a temporary path or a specific path for the uploaded file
        file_path = secure_filename(uploaded_file.filename)
        uploaded_file.save(file_path)

        print('File path of document that will be made into vector store:')
        print(file_path)
        
        # Now, pass the saved file path to ingest_docs function
        ingest_docs(file_path)

        return jsonify({"status": "success", "message": "Document ingested successfully."}), 200
    else:
        return jsonify({"status": "failure", "message": "Invalid file type"}), 400

# Endpoint to generate a response based on a prompt
@app.route('/generate', methods=['POST'])
def api_generate_response():
    # Using request.form to retrieve form-data values
    prompt_message = request.form.get('prompt')

    # Check if prompt is missing
    if not prompt_message:
        return jsonify({"status": "failure", "message": "Prompt is missing from the request body"}), 400

    # Generate response using the provided prompt
    try:
        response = generate_response(prompt_message)
        return jsonify({"status": "success", "response": response}), 200
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error generating response: {str(e)}"}), 500

if __name__ == '__main__':
    print("API Ruinning...")
    app.run(debug=True)
