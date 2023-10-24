# install dependencies:
# pip install python-dotenv Flask flask_restful pdfminer.six langchain dotenv streamlit

from flask import Flask, request, jsonify
import os

# Import your functions here:
from ingest import ingest_docs
from app import generate_response

app = Flask(__name__)

# Endpoint to ingest a document
@app.route('/ingest', methods=['POST'])
def api_ingest():
    # Expect a file in the request and a user's OpenAI API key
    doc_file = request.files['document']
    openai_api_key = request.form['openai_api_key']

    # Save the document temporarily
    doc_file.save("temp_document.pdf")
    
    # Update the .env with the OpenAI API key
    with open(".env", "w") as env_file:
        env_file.write(f"OPENAI_API_KEY={openai_api_key}")
    
    # Call the ingest_docs function
    ingest_docs()

    # Cleanup: Remove the temporary saved file
    os.remove("temp_document.pdf")

    return jsonify({"status": "success", "message": "Document ingested successfully."})

# Endpoint to generate a response based on a prompt
@app.route('/generate', methods=['POST'])
def api_generate_response():
    data = request.json
    prompt_message = data['prompt']

    response = generate_response(prompt_message)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
