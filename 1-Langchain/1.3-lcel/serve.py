from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv

# initial
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# create prompt
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")
])

parser = StrOutputParser()

## create chain
chain = prompt_template|model|parser

## app definition
app = FastAPI(title="LangChain Server",
              version="1.0",
              description="A simple API using Langchain runnable interfaces")

## add chain route
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",port=8000)