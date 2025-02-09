import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
#from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader

# LLM initialization
#llm  = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)



st.set_page_config("Chat with PDF")
st.title("Conversational RAG With PDF")


session_id = st.sidebar.text_input("Session ID")
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files = st.sidebar.file_uploader("Upload PDF file(s).", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    for upload_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(upload_file.getvalue())
            file_name = upload_file.name
        docs = PyPDFLoader(temppdf).load()
        documents.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and latest user question,"
        "which might reference context in chat history, "
        "formulate a standalone question without chat history"
        "Don not answer the question."
        "Just reformulate it if needed else return as it is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
                "You are an assistant for question-answering tasks."
                "Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer, say that you don't know. keep the answer concise."
                "\n\n"
                "{context}"
            )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    user_input = st.text_input("Ask question here")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },
            )
        #st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        #st.write("Chat History:", session_history.messages)

