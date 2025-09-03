import boto3
import json
import os
from datetime import datetime, timezone

s3 = boto3.client("s3")
sagemaker = boto3.client("sagemaker")


def _iso(dt):
    if not dt:
        return None
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)


def handler(event, context):
    """
    Automatically copy trained model to central bucket when training completes
    """
    try:
        # Extract training job details from EventBridge event
        detail = event["detail"]
        training_job_name = detail["TrainingJobName"]
        training_job_status = detail["TrainingJobStatus"]

        print(
            f"Processing training job: {training_job_name}, Status: {training_job_status}"
        )

        if training_job_status != "Completed":
            print("Training job not completed, skipping")
            return

        # Get training job details
        response = sagemaker.describe_training_job(TrainingJobName=training_job_name)
        model_s3_uri = response["ModelArtifacts"]["S3ModelArtifacts"]

        # Parse source S3 location
        source_bucket = model_s3_uri.split("/")[
            2
        ]  # Extract bucket from s3://bucket/path
        source_key = "/".join(model_s3_uri.split("/")[3:])  # Extract key path

        # Central bucket details
        central_bucket = os.environ["CENTRAL_BUCKET"]
        region = os.environ["AWS_REGION"]
        dynamodb_region = os.environ.get("MODEL_REGISTRY_REGION")

        # Create destination key with timestamp and region
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest_key = f"models/{region}/{training_job_name}/{timestamp}/model.tar.gz"

        print(
            f"Copying from {source_bucket}/{source_key} to {central_bucket}/{dest_key}"
        )

        # Copy model to central bucket
        copy_source = {"Bucket": source_bucket, "Key": source_key}

        s3.copy_object(
            CopySource=copy_source,
            Bucket=central_bucket,
            Key=dest_key,
            MetadataDirective="COPY",
            TaggingDirective="COPY",
        )

        # Also create a "latest" version pointer
        latest_key = f"models/{region}/latest/model.tar.gz"
        s3.copy_object(
            CopySource=copy_source,
            Bucket=central_bucket,
            Key=latest_key,
            MetadataDirective="COPY",
            TaggingDirective="COPY",
        )

        # # Add metadata tags
        # s3.put_object_tagging(
        #     Bucket=central_bucket,
        #     Key=dest_key,
        #     Tagging={
        #         'TagSet': [
        #             {'Key': 'TrainingJobName', 'Value': training_job_name},
        #             {'Key': 'Region', 'Value': region},
        #             {'Key': 'Timestamp', 'Value': timestamp},
        #             {'Key': 'ModelType', 'Value': 'federated-learning'}
        #         ]
        #     }
        # )

        print(f"Successfully copied model to central bucket: {dest_key}")

        # Optionally trigger other Lambda functions for model aggregation
        # or notify other regions that a new model is available
        dynamodb = boto3.client("dynamodb", region_name='eu-west-1')

        table_name = os.environ["AGGREGATION_TABLE_NAME"]
        round_number = 2  # Default round number

        # table = dynamodb.Table(table_name)

        dynamodb.put_item(TableName=table_name, 
            Item= {
                "region": {"S": region},        # String type
                "round": {"S": f"{round_number}"},   # Number type (as string)
                "status": {"S": "completed"},   # String type
                "model_s3_uri": {"S": model_s3_uri},
                "central_bucket_uri": {"S": f"s3://{central_bucket}/{dest_key}"},
                "latest_uri": {"S": f"s3://{central_bucket}/{latest_key}"},
                "timestamp": {"S": _iso(datetime.now())},
                "created_at": {"S": _iso(response.get("CreationTime"))},
                "training_start_time": {"S": _iso(response.get("TrainingStartTime"))},
                "training_end_time": {"S": _iso(response.get("TrainingEndTime"))},
                "model_type": {"S": "federated-learning"}
            }
        )


        # Upsert operation - update if exists, create if doesn't exist
        # table.put_item(
        #     Item={
        #         "region": region,
        #         "round": round_number,  # Added required round key
        #         "status": "completed",
        #         "model_s3_uri": model_s3_uri,
        #         "central_bucket_uri": f"s3://{central_bucket}/{dest_key}",
        #         "latest_uri": f"s3://{central_bucket}/{latest_key}",
        #         "timestamp": _iso(datetime.now()),
        #         "created_at": _iso(response.get("CreationTime")),
        #         "training_start_time": _iso(response.get("TrainingStartTime")),
        #         "training_end_time": _iso(response.get("TrainingEndTime")),
        #         "model_type": "federated-learning",
        #     }
        # )

        print(f"Writing aggregation record to DynamoDB table '{table_name}' in {dynamodb_region}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Model copied successfully",
                    "source": model_s3_uri,
                    "destination": f"s3://{central_bucket}/{dest_key}",
                    "latest": f"s3://{central_bucket}/{latest_key}",
                }
            ),
        }

    except Exception as e:
        print(f"Error copying model: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
