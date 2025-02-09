import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage

groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI and chat app
def chatbot_app():
    st.set_page_config(page_title="iChatbot", layout="centered")
    st.title("🧠 AI Chatbot with LangChain")

    # Initialize conversation memory
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory()
    
    # Clear conversation memory
    if st.sidebar.button("Clear Conversation"):
        st.session_state.conversation_memory = ConversationBufferMemory()
        st.sidebar.success("Conversation history cleared!")

    # Initialize LLM model
    llm = ChatGroq(model_name='Gemma2-9b-It', groq_api_key=groq_api_key)

    # Initialize ConversationChain
    conversation_chain = ConversationChain(
                            llm = llm,
                            memory = st.session_state.conversation_memory,
                            prompt = PromptTemplate(
                                    input_variables = ["history", "input"],
                                    template = """ 
                                        You are a helpful assistant who answers questions. 
                                        Here is the conversation so far:
                                        {history}
                                        User: {input}
                                        Assistant:
                                            """ ))

    # User input
    user_question = st.text_input("Please ask question here:")

    # Process input
    if user_question:
        result = conversation_chain.invoke(user_question)
        st.write(result['response'])

        # Add user message
        st.session_state.conversation_memory.chat_memory.add_message(
            ChatMessage(role="human", content=user_question)
        )
        # Add AI response
        st.session_state.conversation_memory.chat_memory.add_message(
            ChatMessage(role="ai", content=result["response"])
        )

    # Display conversation history
    if st.session_state.conversation_memory.chat_memory.messages:
        st.write("### Conversation History")
        for message in st.session_state.conversation_memory.chat_memory.messages:
            if message.type == "human":
                #st.markdown(f"**You:** {message.content}")
                st.markdown(
                    f"""
                    <div style="color: blue; font-weight: bold;">
                        &#128100; <b>You:</b> {message.content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif message.type == "ai":
                #st.markdown(f"**Assistant:** {message.content}")
                st.markdown(
                    f"""
                    <div style="color: green; font-weight: bold;">
                        &#128640; <b>Assistant:</b> {message.content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    chatbot_app()
