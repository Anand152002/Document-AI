# Document-AI

This Streamlit application allows users to upload a PDF document and ask questions about its content. The app uses OpenAI's GPT-3.5 model and Pinecone's vector store for efficient document processing and retrieval, leveraging LangChain for prompt management and parsing.

## Features
PDF Upload: Users can upload any PDF file.
Text Splitting: Splits PDF content into manageable chunks for easier processing.
Question Answering: Queries the PDF's content to answer specific user questions.
Vector Store: Utilizes Pinecone for efficient similarity-based text retrieval.

## Getting Started
1) Clone the repository:

 ```bash

https://github.com/Anand152002/Document-AI.git
 ```

2)Install dependencies:
 ```bash
pip install -r requirements.txt
 ```
3)API Keys Configuration:

Create an environment file named key_api.py and add your OpenAI and Pinecone API keys as follows:
 ```bash
openai_api_key = "YOUR_OPENAI_API_KEY"
pine_api_key = "YOUR_PINECONE_API_KEY"
 ```

4)Running the App
 ```bash

streamlit run app.py
 ```
