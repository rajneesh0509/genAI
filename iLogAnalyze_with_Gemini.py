import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

os.getenv("GOOGLE_API_KEY")

st.set_page_config("iLogAnalyze")
st.header("Logfile query using LLM and VectorDB")

def log_to_text(logfile):
    converted_logfile = logfile.read().decode("utf-8")
    return converted_logfile

def text_to_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunk = text_splitter.split_text(text)
    return chunk

def get_vector_store(chunk):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorDB = FAISS.from_texts(chunk, embeddings)
    vectorDB.save_local("faiss_index")

def conversation_chain():
    #llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chat_template = """ 
    Please answer the below question based on provided context only. Answer should be  in as much details as possible.
    If the answer is not available in given context, just say "Answer is not available in the context." 
    Context:\n {context} \n
    Question:\n {question} \n
    """
    prompt = PromptTemplate(template=chat_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorDB = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vectorDB.similarity_search(question)
    chain = conversation_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    print(response)
    st.write(response["output_text"])

def main():
    #st.set_page_config("iLog")
    #st.header("Logfile query using LLM and VectorDB")
    question = st.text_input("Please enter your question from logfile")
    if question:
        user_input(question)
    with st.sidebar:
        logfile = st.file_uploader("Please upload a logfile and click submit", type= ["text", "log"], accept_multiple_files=False)
        if st.button("Submit"):
            text = log_to_text(logfile)
            chunk = text_to_chunk(text)
            get_vector_store(chunk)
            st.success("Done")


if __name__ == "__main__":
    main()


