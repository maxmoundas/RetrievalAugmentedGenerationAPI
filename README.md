# Retrieval Augmented Generation API

This API serves as a bridge to perform retrieval augmented generation (RAG) on a set of ingested documents. Using the OpenAI GPT-4 model for generating text, it will first retrieve the most relevant parts of the ingested documents and then generate a comprehensive response based on the provided prompt.

## Setup
1. Install Dependencies
```
pip install python-dotenv Flask flask_restful pdfminer.six langchain streamlit plotly openai matplotlib pandas scipy scikit-learn tiktoken faiss-cpu werkzeug
```
Note: replace faiss-cpu with faiss-gpu if you would like to use your GPU. You must have CUDA 11.4 for a successful install. Reference (https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for specific installation instructions.

2. Run the application:
```
python api.py
```
This will start the Flask server on the default port (5000).

## API Endpoints
1. Collect OpenAI API key and save it to .env:
   * Endpoint: /collect_key
   * Type: POST
   * URL: http://127.0.0.1:5000/collect_key
   * Body (type: form-data): 
      * Key: openai_api_key
      * Value: Your OpenAI API Key
   * Description: Stores the OpenAI API key securely for the current session.
2. Ingest a document:
   * Endpoint: /ingest
   * Type: POST
   * URL: http://127.0.0.1:5000/ingest
   * Body (type: form-data): 
      * Key: prompt
      * Value: Prompt you would like to use on the document
   * Description: Ingests and vectorizes a document for similarity search. Supports .txt, .pdf, and .docx files.
3. Generate a response:
   * Type: POST
   * URL: http://127.0.0.1:5000/generate
   * Body (type: form-data): 
      * Key: prompt
      * Value: Prompt you would like to use on the document
   * Description: Generates a comprehensive response using the ingested documents based on the provided prompt.

## Notes
* Ensure the provided documents are in a format supported by the API (.txt, .pdf, .docx).

## Future Improvements:
* Implement authentication to enhance security.
* Expand the types of documents supported by the system.
* Improve embedding practices to improve output.
* Improve storage of documents to improve space efficiency