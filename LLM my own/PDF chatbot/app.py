from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceEndpoint

import langchain
langchain.verbose = False

# Load environment variables
load_dotenv()

# Define the process of extracting text from a PDF
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text
# Define the process of processing PDF text
def pdf_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)
    #docs = text_splitter.create_documents(text)
    print(chunks)

    # Convert text into word embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    print(db)

    return db

def main():
    st.title('_Chat with my_ :blue[PDF] :')
    pdf = st.file_uploader("Upload Your PDF File ", type="pdf")

    if pdf is not None:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(pdf)
        print(text)

        # Process the extracted text
        knowledge_base = pdf_text(text)
        print(knowledge_base)

        query = st.text_input("Ask a question from the PDF")
        cancel_button = st.button("Cancel")

        if cancel_button:
            st.stop()
        if query:
            docs = knowledge_base.similarity_search(query)
            llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
            chain = load_qa_chain(llm, chain_type="stuff")

            response = chain.invoke(input={"question": query, "input_document": docs})
            
            st.write(response["Output Text"])

if __name__ == "__main__":
    main()
