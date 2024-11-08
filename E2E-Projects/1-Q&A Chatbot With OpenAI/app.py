import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chatbot With OPENAI"
api_key = os.getenv("OPENAI_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the use queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

## Title of the app
st.title("Q & A Chatbot With OpenAI")

## Sidebar for setting
st.sidebar.title("Settings")

## Dropdown to select OpenAI models
llm = st.sidebar.selectbox("Select an Open AI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")