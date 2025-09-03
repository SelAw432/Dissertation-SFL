# New Update
import os
import time
import boto3
import json
from datetime import datetime, timedelta
import re

s3 = boto3.client("s3")
sagemaker = boto3.client("sagemaker")
sagemaker_runtime = boto3.client("sagemaker-runtime")
dynamodb = boto3.resource("dynamodb", region_name="eu-west-1")


def handler(event, context):
    """
    Lambda function for real-time inference that automatically manages endpoints
    """
    try:
        # Get environment variables
        LOCAL_BUCKET = os.environ.get("LOCAL_BUCKET")
        role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
        instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge")
        region = os.environ.get("AWS_REGION")

        if region == "eu-west-1":
            training_image = "763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"
        elif region == "us-west-1":
            training_image = os.environ.get("TRAINING_IMAGE_US_WEST_1")
        else:
            training_image = os.environ.get("TRAINING_IMAGE_US_WEST_2")

        # Extract input data from event
        input_data = event.get("data", event.get("body", {}))
        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        # Get or create endpoint
        endpoint_name = get_or_create_endpoint(
            LOCAL_BUCKET, role_arn, training_image, instance_type, region
        )

        # Make prediction
        prediction = make_prediction(endpoint_name, input_data)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "prediction": prediction,
                    "endpoint": endpoint_name,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
        }

    except Exception as e:
        print(f"Inference error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def get_or_create_endpoint(
    LOCAL_BUCKET, role_arn, training_image, instance_type, region
):
    """
    Get existing endpoint or create new one with latest model
    """
    endpoint_name = f"federated-inference-{region}"

    try:
        # Check if endpoint exists and is in service
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = response["EndpointStatus"]

        if endpoint_status == "InService":
            print(f"Using existing endpoint: {endpoint_name}")
            return endpoint_name
        elif endpoint_status in ["Creating", "Updating"]:
            print(f"Endpoint {endpoint_name} is {endpoint_status}, waiting...")
            wait_for_endpoint(endpoint_name)
            return endpoint_name
        else:
            print(f"Endpoint {endpoint_name} status: {endpoint_status}, recreating...")
            # Delete and recreate if in failed state
            cleanup_endpoint_resources(endpoint_name)

    except sagemaker.exceptions.ClientError as e:
        if "ValidationException" in str(e):
            print(f"Endpoint {endpoint_name} does not exist, creating new one...")
        else:
            raise e

    # Create new endpoint with latest model
    return create_new_endpoint(
        endpoint_name, LOCAL_BUCKET, role_arn, training_image, instance_type, region
    )


def get_model_from_training_job(training_job_name):
    """
    Get model URI from a completed training job
    """
    try:
        response = sagemaker.describe_training_job(TrainingJobName=training_job_name)
        model_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
        print(f"Got model from training job {training_job_name}: {model_uri}")
        return model_uri
    except Exception as e:
        print(f"Failed to get model from training job {training_job_name}: {str(e)}")
        return None


def get_latest_model_from_s3(LOCAL_BUCKET):
    """
    Find the latest trained model in S3 based on your training job structure
    """
    try:
        print(f"Searching for models in bucket: {LOCAL_BUCKET}")

        # List all objects in models/ prefix to find the most recent model.tar.gz
        response = s3.list_objects_v2(Bucket=LOCAL_BUCKET, Prefix="models/")

        if "Contents" not in response:
            raise Exception("No objects found in models/ prefix")

        # Filter for model.tar.gz files and sort by last modified
        model_files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.endswith("/output/model.tar.gz"):
                model_files.append({"key": key, "last_modified": obj["LastModified"]})

        if not model_files:
            raise Exception("No model.tar.gz files found")

        # Sort by last modified date (most recent first)
        model_files.sort(key=lambda x: x["last_modified"], reverse=True)
        latest_model_key = model_files[0]["key"]

        model_uri = f"s3://{LOCAL_BUCKET}/{latest_model_key}"
        print(f"Found latest model: {model_uri}")
        print(f"Last modified: {model_files[0]['last_modified']}")

        return model_uri

    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Failed to find latest model: {str(e)}")


def create_new_endpoint(
    endpoint_name, LOCAL_BUCKET, role_arn, training_image, instance_type, region
):
    """
    Create a new endpoint with the latest model
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"federated-model-{region}-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    try:
        # First try to get model from latest training job
        model_uri = None

        # Option 1: Try to find the latest completed training job
        try:
            training_jobs = sagemaker.list_training_jobs(
                StatusEquals="Completed",
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=10,
            )

            for job in training_jobs["TrainingJobSummaries"]:
                job_name = job["TrainingJobName"]
                if job_name.startswith("federated-training"):
                    model_uri = get_model_from_training_job(job_name)
                    if model_uri:
                        print(f"Using model from training job: {job_name}")
                        break
        except Exception as e:
            print(f"Could not find training jobs: {str(e)}")

        # Option 2: Fall back to S3 search if no training job found
        if not model_uri:
            print("No recent training job found, searching S3...")
            model_uri = get_latest_model_from_s3(LOCAL_BUCKET)

        # Create SageMaker model
        print(f"Creating model: {model_name}")
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": training_image,
                "ModelDataUrl": model_uri,
                "Mode": "SingleModel",
            },
            ExecutionRoleArn=role_arn,
        )

        # Create endpoint configuration
        print(f"Creating endpoint config: {endpoint_config_name}")
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )

        # Create endpoint
        print(f"Creating endpoint: {endpoint_name}")
        sagemaker.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )

        # Wait for endpoint to be ready
        wait_for_endpoint(endpoint_name)

        return endpoint_name

    except Exception as e:
        print(f"Failed to create endpoint: {str(e)}")
        # Cleanup partial resources
        cleanup_partial_resources(model_name, endpoint_config_name, endpoint_name)
        raise


def wait_for_endpoint(endpoint_name, max_wait_time=900):  # 15 minutes max wait
    """
    Wait for endpoint to be InService
    """
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            if status == "InService":
                print(f"Endpoint {endpoint_name} is ready!")
                return
            elif status == "Failed":
                raise Exception(f"Endpoint {endpoint_name} failed to deploy")
            else:
                print(f"Endpoint {endpoint_name} status: {status}, waiting...")
                time.sleep(30)  # Wait 30 seconds between checks

        except Exception as e:
            if "ValidationException" not in str(e):
                raise e
            time.sleep(30)

    raise Exception(
        f"Endpoint {endpoint_name} did not become ready within {max_wait_time} seconds"
    )


def make_prediction(endpoint_name, input_data):
    """
    Make prediction using the endpoint
    """
    try:
        # Convert input to CSV format for XGBoost
        if isinstance(input_data, dict):
            # Convert dict values to CSV (adjust order based on your features)
            features = list(input_data.values())
            csv_input = ",".join(map(str, features))
        elif isinstance(input_data, list):
            csv_input = ",".join(map(str, input_data))
        else:
            csv_input = str(input_data)

        # Invoke endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name, ContentType="text/csv", Body=csv_input
        )

        # Parse response
        result = response["Body"].read().decode("utf-8")
        prediction = float(result.strip())

        return prediction

    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def cleanup_endpoint_resources(endpoint_name):
    """
    Clean up endpoint and related resources
    """
    try:
        # Delete endpoint
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleted endpoint: {endpoint_name}")

        # Find and delete old endpoint configs (keep last 2)
        response = sagemaker.list_endpoint_configs(
            NameContains=endpoint_name, MaxResults=10
        )

        configs = sorted(
            response["EndpointConfigs"], key=lambda x: x["CreationTime"], reverse=True
        )

        # Delete old configs (keep 2 most recent)
        for config in configs[2:]:
            try:
                sagemaker.delete_endpoint_config(
                    EndpointConfigName=config["EndpointConfigName"]
                )
                print(f"Deleted old config: {config['EndpointConfigName']}")
            except:
                pass

    except Exception as e:
        print(f"Cleanup error: {str(e)}")


def cleanup_partial_resources(model_name, endpoint_config_name, endpoint_name):
    """
    Cleanup resources if endpoint creation fails
    """
    try:
        # Try to delete in reverse order
        try:
            sagemaker.delete_endpoint(EndpointName=endpoint_name)
        except:
            pass

        try:
            sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        except:
            pass

        try:
            sagemaker.delete_model(ModelName=model_name)
        except:
            pass

    except Exception as e:
        print(f"Partial cleanup error: {str(e)}")


# Lambda function to update endpoint with new model (optional)
def update_endpoint_handler(event, context):
    """
    Separate Lambda to update endpoint with new model after training
    """
    try:
        training_job_name = event.get("training_job_name")
        if not training_job_name:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "training_job_name required"}),
            }

        LOCAL_BUCKET = os.environ.get("LOCAL_BUCKET")
        role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
        training_image = os.environ.get("TRAINING_IMAGE")
        region = os.environ.get("AWS_REGION")

        endpoint_name = f"federated-inference-{region}"

        # Update endpoint with new model
        update_endpoint_with_new_model(
            endpoint_name,
            training_job_name,
            LOCAL_BUCKET,
            role_arn,
            training_image,
            region,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {"message": "Endpoint update initiated", "endpoint_name": endpoint_name}
            ),
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def update_endpoint_with_new_model(
    endpoint_name, training_job_name, LOCAL_BUCKET, role_arn, training_image, region
):
    """
    Update existing endpoint with newly trained model
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"federated-model-{region}-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    # Get training job output
    training_job = sagemaker.describe_training_job(TrainingJobName=training_job_name)
    model_uri = training_job["ModelArtifacts"]["S3ModelArtifacts"]

    # Create new model
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": training_image,
            "ModelDataUrl": model_uri,
            "Mode": "SingleModel",
        },
        ExecutionRoleArn=role_arn,
    )

    # Create new endpoint config
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": os.environ.get("INSTANCE_TYPE", "ml.m5.xlarge"),
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # Update endpoint
    sagemaker.update_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
