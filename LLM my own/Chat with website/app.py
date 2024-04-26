import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain_core.prompts import   ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



   
    

def vector_store(url):
    loader = WebBaseLoader(url)
    doc = loader.load()
    chunk_size = 500
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_chunks = text_splitter.split_documents(doc)

    embeddings = HuggingFaceEmbeddings()
    
    db = FAISS.from_documents(doc_chunks, embeddings)

    return db

def  retrive_chain(db):
    llm=  HuggingFaceEndpoint(
    #repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    repo_id="deepset/roberta-large-squad2")
    retriever = db.as_retriever(search_kwargs={'k': 1})
    chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="chat_history"), 
     ("user","{input}"),
     ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
     ])
    
    chat_retriever_chain = create_history_aware_retriever(
    llm, retriever, chat_prompt )

    return chat_retriever_chain

def convertional_rag(chat_retriever_chain):
    llm=  HuggingFaceEndpoint(repo_id="deepset/roberta-large-squad2")
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(chat_retriever_chain, stuff_documents_chain)

def get_response( user_query):
   chat_retriever_chain = retrive_chain(st.session_state.db)
   #convertional_rag = convertional_rag(chat_retriever_chain)
   response = convertional_rag(chat_retriever_chain).invoke(
            {"input": user_query, "chat_history":st.session_state.chat_history }
        )
   return response["answer"]
    


st.title("Chat with website 🌍🌍🌍🌍🌍")
load_dotenv()


with st.sidebar:
    st.header("Setting")
    url = st.text_input("Website url")
if url is None or url =="":
    st.info("Please enter website")
else:
   if 'chat_history' not in st.session_state:

        st.session_state.chat_history  = [
            AIMessage(content="Hello 👋 How can I help you?"),
        ]
   if 'db' not in st.session_state:
       st.session_state.db = vector_store(url)
   
  
   user_query = st.chat_input("Type text.....")
   if user_query is not None and user_query!="":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response ))

        # retrive_doc = chat_retriever_chain.invoke(
        #     {"input": user_query, "chat_history":st.session_state.chat_history }
        # )

        



   for message in st.session_state.chat_history :
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
    
