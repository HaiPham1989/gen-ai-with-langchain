import boto3
import json
import os
from dotenv import load_dotenv
load_dotenv()

prompt_data = """
Act as Shakespaere and write poem on machine Learning
"""

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name='us-east-1',  # e.g., 'us-west-2'
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

payload = {
    "prompt":prompt_data,
    "max_gen_len":512,
    "top_p":0.9,
    "temperature":0.6 
}

try:
    # Make the request to Bedrock
    response = bedrock_client.invoke_model(
        body=json.dumps(payload),
        modelId="meta.llama3-8b-instruct-v1:0",
        contentType="application/json",
        accept="application/json",
    )
    
    # Parse the response
    response_body = json.loads(response.get("body").read())
    response_text = response_body['generation']
    print("Response from Bedrock:", response_text)

except Exception as e:
    print("Error invoking the Bedrock model:", str(e))