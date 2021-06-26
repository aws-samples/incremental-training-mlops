import os
import io
import boto3
import json
import csv
import base64
import subprocess
import sys 
import uuid 
import traceback

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']


runtime= boto3.client('runtime.sagemaker')



def lambda_handler(event, context):
    body = event["content"]
    payload = base64.b64decode(body)
    runtime_client = boto3.client('runtime.sagemaker')
    response = runtime_client.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                  ContentType='application/octet-stream', 
                                  Body=payload)
    
    result = response['Body'].read().decode('utf-8')    
    return json.loads(result)