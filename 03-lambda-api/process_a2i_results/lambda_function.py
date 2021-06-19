import json
import boto3 
from model import get_latest_model_path
from datetime import datetime
import re
import os
from prepare_data import convert_a2i_to_augmented_manifest
from urllib.parse import urlparse


BUCKET = os.environ['BUCKET']
PIPELINE = os.environ['PIPELINE']
MODEL_GROUP = os.environ['MODEL_GROUP']


body = "body"
detail = "detail"
humanLoopName = "humanLoopName"
s3_path = "s3-image-path"
loop_status = "HumanLoopStatus"
string_value = "stringValue"


a2i = boto3.client('sagemaker-a2i-runtime')
sm_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
 
completed_human_loops = []     



def lambda_handler(event, context):
    # TODO implement
    print(json.dumps(event)) 
    records = event['Records']
    for record in records: 
        if body in record: 
            bodyjson = json.loads(record[body]) 
            if detail in bodyjson:
                resp = a2i.describe_human_loop(HumanLoopName=bodyjson[detail][humanLoopName])
                if resp[loop_status] == "Completed":
                    completed_human_loops.append(resp)
    if len(completed_human_loops)>0:
        output=[]
        training_file = 'meta_train.csv'
        path = "/tmp/{}".format(training_file)
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        
        prefix = "a2i-result/{}".format(str(timestamp))
        print(prefix)
        with open(path, 'w') as outfile:        
            outfile.write("Filename,Label,Remark\n")
            for resp in completed_human_loops:
                splitted_string = re.split('s3://' +  BUCKET + '/', resp['HumanLoopOutput']['OutputS3Uri'])
                output_bucket_key = splitted_string[1]
        
                response = s3_client.get_object(Bucket=BUCKET, Key=output_bucket_key)
                content = response["Body"].read()
                json_output = json.loads(content)
                print(json_output)
                # convert using the function
                augmented_manifest, s3_path = convert_a2i_to_augmented_manifest(json_output)
                o = urlparse(s3_path, allow_fragments=False)
                obucket = o.netloc 
                okey = o.path
                of = okey.split('/')[-1]
                copy_source = {
                    'Bucket':obucket, 
                    'Key': okey[1:]
                }
                tbucket = s3.Bucket(BUCKET)
                print(copy_source, "{}/train/{}".format(prefix, of))
                tbucket.copy(copy_source, "{}/train/{}".format(prefix, of))
                
    
                outfile.write(augmented_manifest)
                outfile.write('\n')            
        
        
        
        
        s3_client.upload_file(path, Bucket=BUCKET, Key="{}/{}".format(prefix, training_file))                
        s3_path = "s3://{}/{}".format(BUCKET, prefix)
        last_model_path = get_latest_model_path(MODEL_GROUP) 
        parameters = [
            {
                'Name':'TrainData',
                'Value': s3_path
            },
    
            {
                'Name':'ValidationData',
                'Value': s3_path
            },
    
            {
                'Name':'ModelData',
                'Value': last_model_path
            },
    
        ]
    
        response = sm_client.start_pipeline_execution( PipelineName = PIPELINE, PipelineParameters=parameters)
        
    return {
        'statusCode': 200,
        # 'body': json.dumps(completed_human_loops)
        'body': 'finished'
    }
