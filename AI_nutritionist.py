import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

os.getenv("GOOGLE_API_KEY")

def image_process(image_file):
    if image_file is not None:
        data_bytes = image_file.getvalue()
        image_parts = [
            {
                "mime_type": image_file.type,
                "data": data_bytes
            }
        ]
        return image_parts
    else:
        print("Please upload image file.")

prompt_template = """ 
You are an expert Nutritionist. Please check uploaded food image.
As per food items in the image, calculate total calories and provide 
details of every food item with calories intake in the below format.

1. Item 1 - No. of calories
2. Item 2 - No. of calories
3. Item 3 - No. of calories
---------------------------
---------------------------

Input prompt: {user_input}
Image: {image_data}
"""

def llm_response(user_input, image_data):
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_input"])   #.format(user_input=user_input)
    llm = GoogleGenerativeAI(model="gemini-1.5-flash")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    llm_response = llm_chain.invoke({"user_input": user_input, "image_data": image_data[0]})
    return llm_response

st.set_page_config(page_title="Health calorie check")
st.header("Health calorie check App")
user_input = st.text_input("Input prompt", key="input")

uploaded_image = st.file_uploader("Please upload image", type=["jpg", "jpeg", "png"])
# Display uploaded image
image = ""
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded image", use_column_width=True)

if st.button("Tell me total calories description"):
    image_data = image_process(uploaded_image)
    response = llm_response(user_input, image_data)
    st.write(response)

    