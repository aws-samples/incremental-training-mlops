# tomofun_competition_template


### Architecture 

![architecture](./architecture.jpg)

1. Training model by customized container  
2. Deploying model
3. Triggering endpoint by Lambda function 
4. Integrating Lambda function and API-Gateway 
5. Saving posted audio files to S3 
6. Initiating A2I tasks 
7. Users labels the incoming audios 
8. EventBridge passing label complete event to SQS 
9. User trigger retraining / update model  


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

