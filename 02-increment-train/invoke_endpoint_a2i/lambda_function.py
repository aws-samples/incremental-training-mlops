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
A2IFLOW_DEF = os.environ['A2IFLOW_DEF']
BUCKET = os.environ['BUCKET']
KEY = os.environ['KEY']

runtime= boto3.client('runtime.sagemaker')
s3_client = boto3.client('s3')
a2i = boto3.client('sagemaker-a2i-runtime')

def object_with_max_prob(dets):
    dets = dets["probability"]
    dets = sorted(dets,reverse=True)
    return dets[0]  


def lambda_handler(event, context):
    body = event["content"]
    payload = base64.b64decode(body)
    runtime_client = boto3.client('runtime.sagemaker')
    response = runtime_client.invoke_endpoint(EndpointName=ENDPOINT_NAME, 
                                  ContentType='application/octet-stream', 
                                  Body=payload)
    
    result = response['Body'].read().decode('utf-8')
    print(result)
    prob = object_with_max_prob(json.loads(result))
    
    if prob < 1: 
        task = str(uuid.uuid4())
        file_name = "{}.wav".format(task) 
                    
        s3_client.put_object(Body=payload, Bucket=BUCKET, Key="{}/{}".format(KEY,file_name))
        s3_filename = "s3://{}/{}/{}".format(BUCKET, KEY, file_name)                    
        inputContent = {
            "initialValue": prob,
            "taskObject": s3_filename # the s3 object will be passed to the worker task UI to render
        }
        # start an a2i human review loop with an input
        flowDefinitionArn = A2IFLOW_DEF                   
        try: 
            start_loop_response = a2i.start_human_loop(
                HumanLoopName=task,
                FlowDefinitionArn=flowDefinitionArn,
                HumanLoopInput={
                    "InputContent": json.dumps(inputContent)
                }
            )
            a2i_arn = start_loop_response['HumanLoopArn'].split('/')[-1]
        except:
            traceback.print_exc()

        # https://forums.aws.amazon.com/thread.jspa?messageID=961211&tstart=0    
    return json.loads(result)