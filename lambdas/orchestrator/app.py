import os, time, boto3, json
from datetime import datetime

s3 = boto3.client("s3")
sagemaker = boto3.client("sagemaker")
dynamodb = boto3.resource("dynamodb")

def handler(event, context):

    region = os.environ.get("AWS_REGION")
    bucket_name = os.environ.get("LOCAL_BUCKET")
    role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge")

    if region == "eu-west-1":
        training_image = "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"
    elif region == "us-west-1":
        training_image = "763104351884.dkr.ecr.us-west-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"
    else:
        training_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"

    # Get latest global model version from S3
    # This is simplified - you'd implement proper versioning logic

    timestamp = datetime.now().strftime("%H%M%S")
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    job_name = f"federated-training-{region}-{timestamp}"

    # Build dynamic S3 output path: date folder → job folder
    s3_output = f"s3://{bucket_name}/models/{date_str}/"

    # Get hyperparameters from event or use defaults
    epochs = "100"
    batch_size = "64"
    learning_rate = "0.001"
    hidden_layers = "128,64,32"
    dropout = "0.2"

    # Create SageMaker training job
    try:
        response = sagemaker.create_training_job(
            TrainingJobName=job_name,
            RoleArn=role_arn,
            AlgorithmSpecification={
                "TrainingImage": training_image,
                "TrainingInputMode": "File",
            },
            InputDataConfig=[
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{bucket_name}/data/training/",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                }
            ],
            OutputDataConfig={"S3OutputPath": s3_output},
            ResourceConfig={
                "InstanceType": instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 3600},
            # this is the part
            HyperParameters={
                "sagemaker_program": "sagemaker_train.py",
                "sagemaker_submit_directory": f"s3://{bucket_name}/code/sourcedir.tar.gz",
                "batch-size": "64",
                "dropout": "0.2",
                "epochs": "100",
                "hidden-layers": "128,64,32",
                "learning-rate": "0.001",
            },
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Training job started",
                    "jobName": job_name,
                    "output_path": s3_output,
                    "hyperparameters": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "hidden_layers": hidden_layers,
                        "dropout": dropout,
                    },
                }
            ),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
