import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
#from langchain_community.llms import ollama
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Q&A chatbot with Ollama"
#LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a good chat assistant. Please answer below query."),
        ("user", "Question: {question}")
    ]
)
llm  = st.sidebar.selectbox("Select open source model", ["llama3:8b-instruct-q4_K_M"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens =st.sidebar.slider("Max_tokens", min_value=50, max_value=300, value=150)

def generate_response(question, llm=llm, temperature=temperature, max_tokens=max_tokens):
    llm = ChatOllama(model=llm, temperature=temperature, max_tokens=max_tokens)
    out_parser = StrOutputParser()
    chain = prompt | llm | out_parser
    answer = chain.invoke({"question": question})
    return answer

st.title("Q&A chatbot using Ollama")

st.write("Please ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide user input.")