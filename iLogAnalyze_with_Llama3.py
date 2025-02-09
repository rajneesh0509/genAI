import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.getenv("GOOGLE_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)



st.set_page_config("iLogAnalyze")
st.header("Logfile query using LLM and VectorDB")

session_id = st.sidebar.text_input("Session ID")
if 'store' not in st.session_state:
    st.session_state.store={}

def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]


def log_to_text(logfile):
    converted_logfile = logfile.read().decode("utf-8")
    return converted_logfile

def text_to_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunk = text_splitter.split_text(text)
    return chunk

def get_vector_store(chunk):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorDB = FAISS.from_texts(chunk, embeddings)
    #vectorDB.save_local("faiss_index")
    return vectorDB.as_retriever()


logfile = st.sidebar.file_uploader("Please upload a logfile and click submit", type= ["text", "log"], accept_multiple_files=False)
if st.sidebar.button("Submit"):
    text = log_to_text(logfile)
    chunk = text_to_chunk(text)
    retriever = get_vector_store(chunk)
    st.sidebar.success("VectorDB created")
    
text = log_to_text(logfile)
chunk = text_to_chunk(text)
retriever = get_vector_store(chunk)
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)


def conversation_chain():
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chat_template = """ 
    Please answer the below question based on provided context only. Answer should be  in as much details as possible.
    If the answer is not available in given context, just say "Answer is not available in the context." 
    Context:\n {context} \n
    Question:\n {question} \n
    """
    prompt = PromptTemplate(template=chat_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain


def get_rag_chain(llm, retriever):
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
                "You are a helpful assistant for question-answering tasks."
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
    return rag_chain


rag_chain = get_rag_chain(llm, retriever)

conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )



user_input = st.text_input("Ask question from logfile here")
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
    st.write("Chat History:", session_history.messages)

