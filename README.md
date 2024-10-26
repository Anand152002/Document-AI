# Document-AI

This Streamlit application allows users to upload a PDF document and ask questions about its content. The app uses OpenAI's GPT-3.5 model and Pinecone's vector store for efficient document processing and retrieval, leveraging LangChain for prompt management and parsing.

Features
PDF Upload: Users can upload any PDF file.
Text Splitting: Splits PDF content into manageable chunks for easier processing.
Question Answering: Queries the PDF's content to answer specific user questions.
Vector Store: Utilizes Pinecone for efficient similarity-based text retrieval.
Getting Started
Prerequisites
Python 3.8 or later
Streamlit
OpenAI API Key: Required to access OpenAI's GPT-3.5 model.
Pinecone API Key: Required to use Pinecone for document retrieval.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/pdf-question-answering-app.git
cd pdf-question-answering-app
Install dependencies:

bash
Copy code
pip install -r requirements.txt
API Keys Configuration:
Create a file named key_api.py and add your OpenAI and Pinecone API keys as follows:

python
Copy code
openai_api_key = "YOUR_OPENAI_API_KEY"
pine_api_key = "YOUR_PINECONE_API_KEY"
Running the App
Run the Streamlit app locally by executing:

bash
Copy code
streamlit run app.py
Usage
Upload a PDF document using the file uploader.
Type a question in the text box and click "Ask".
The app will display an answer based on the PDF content.
File Structure
app.py: Main application file.
key_api.py: File to store API keys securely.
requirements.txt: Dependencies required to run the application.
