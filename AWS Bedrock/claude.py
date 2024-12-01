import boto3
import json
import os
from dotenv import load_dotenv
load_dotenv()

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

messages = [
    {
        "role": "user",
        "content": prompt_data
    }
]

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1',  # e.g., 'us-west-2'
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

payload={
    "messages":messages,
    "max_tokens":512,
    "temperature":0.8,
    "top_p":0.8
}
body = json.dumps(payload)
model_id = "ai21.jamba-1-5-large-v1:0"
response = bedrock_client.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
#print(response_body)
response_text = response_body.get("choices")[0].get("message").get("content")
print(response_text)