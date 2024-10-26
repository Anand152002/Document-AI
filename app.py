import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set OpenAI and Pinecone API keys
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pine_api_key

# Initialize OpenAI chat model
model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# Define the chat prompt template
template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()

# Function to process PDF and question
def process_pdf_and_question(pdf_file, question):
    # Read PDF file
    pdf_reader = PdfReader(pdf_file)
    raw_text = ''
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_text(raw_text)

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=())

    # Initialize Pinecone
    pc = Pinecone(api_key=pine_api_key)
    index_name = "my-pdf-index"
    index = pc.Index(index_name)

    # Create Pinecone vector store from documents
    pinecone = PineconeVectorStore.from_texts(
        documents, embeddings, index_name=index_name
    )

    # Create the processing chain
    chain = (
        {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    # Invoke the processing chain with the question
    return chain.invoke(question)

# Streamlit UI
st.title("PDF Question Answering")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    # Text input for question
    question = st.text_input("Enter your question")
    if st.button("Ask"):
        # Process PDF and question
        result = process_pdf_and_question(uploaded_file, question)
        # Display the result
        st.write("Answer:", result)
