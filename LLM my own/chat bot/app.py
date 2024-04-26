import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader

# st.title("HuggingFace Open Source Chat Bot")
embeddings = HuggingFaceEmbeddings()
load_dotenv()
st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()

db = FAISS.from_documents(data, embeddings)

def get_text():
    input_text = st.text_input("You: ", key= input)
    return input_text


user_input=get_text()
submit = st.button('Find similar Things')  

if submit:
    
    #If the button is clicked, the below snippet will fetch us the similar text
    docs = db.similarity_search(user_input,k=3)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)