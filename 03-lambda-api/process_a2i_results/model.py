import boto3 


sm_client = boto3.client('sagemaker')

def get_latest_model_path(model_group):
    info = sm_client.list_model_packages(ModelPackageGroupName=model_group)
    models = info['ModelPackageSummaryList']
    latest = sorted(models, key=lambda x:x['ModelPackageVersion'])[-1]
    latestarn = latest['ModelPackageArn']
    resp = sm_client.describe_model_package(ModelPackageName=latestarn)
    s3_path = resp['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    return s3_path

