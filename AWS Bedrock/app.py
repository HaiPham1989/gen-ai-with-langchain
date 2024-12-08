import json
import os
from dotenv import load_dotenv
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model to generate Embedding
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader

## Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLM Model
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


## Bedrock Clients
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1',  # e.g., 'us-west-2'
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

## Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")
    
def get_claude_llm():
    ## create the Anthropic model
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock,
        model_kwargs={'max_tokens': 512})
    
    return llm

def get_llama2_llm():
    ## create the Anthropic model
    llm = Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={'max_gen_len': 512})
    
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question
at the end but use atleast summarize with 250 words with detailed explanation.
If you don't know the answer, just say that you don't know, don't try to make up an answer
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

def get_response_llm(llm,vectorstore,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3},  
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")
    load_dotenv()
        
    user_question = st.text_input("Ask a Question from the PDF files")
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")
                
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings)
            llm = get_claude_llm()
            
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")
            
if __name__ == "__main__":
    main()