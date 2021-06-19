import boto3 

import argparse
import time


# Parse argument variables passed via the DeployModel processing step
parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str)
parser.add_argument('--endpoint-name', type=str)
parser.add_argument('--region', type=str)
args = parser.parse_args()

timestamp = str(int(time.time())) 
model_name = args.model_name
endpoint_name = args.endpoint_name 
region = args.region 

endpoint_config_name = "audio-vgg16-modelconfig-"+ timestamp



sm_client = boto3.client('sagemaker', region)



def create_endpoint_config(endpoint_config_name, model_name):
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName = endpoint_config_name,
        ProductionVariants=[{
            'InstanceType':'ml.m5.xlarge',
            'InitialVariantWeight':1,
            'InitialInstanceCount':1,
            'ModelName':model_name,
            'VariantName':'AllTraffic'}])


def update_endpoint(endpoint_name, endpoint_config_name): 
    print("EndpointName={}".format(endpoint_name))

    create_endpoint_response = sm_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name)
    print(create_endpoint_response['EndpointArn'])
    

create_endpoint_config(endpoint_config_name, model_name)
update_endpoint(endpoint_name, endpoint_config_name)

