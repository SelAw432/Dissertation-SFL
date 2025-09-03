import json
import boto3
import os
import tempfile
import tarfile
import numpy as np
import shutil
import torch
from datetime import datetime, timezone

s3 = boto3.client('s3')

def _iso(dt):
    if not dt:
        return None
    if isinstance(dt, datetime):
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)

def handler(event, context):
    
    dynamodb = boto3.client('dynamodb', region_name='eu-west-1')
    tableName = 'fl-aggregationqueue'

    # Scan all jobs to check completion status
    response = dynamodb.scan(
        TableName=tableName
    )

    # Filter completed jobs and sort by completion time (most recent first)
    completed_jobs = [
        job for job in response['Items'] 
        if job.get('status', {}).get('S') == 'completed'
    ]
    
    # Sort by completion time (most recent first)
    completed_jobs.sort(
        key=lambda job: datetime.fromisoformat(job.get('training_end_time', {}).get('S','')), 
        reverse=True
    )
    
    # Check if we have at least 2 completed jobs
    if len(completed_jobs) > 1:
        last_two_jobs = completed_jobs[:2]

        # Get the S3 bucket and key for the most recent completed job
        model_1 = last_two_jobs[0].get('central_bucket_uri', {}).get('S','')

        # Get the S3 bucket and key for the most recent completed job
        model_2 = last_two_jobs[1].get('central_bucket_uri', {}).get('S','')

        # Use the region from the most recent completed job
        region_1 = last_two_jobs[0].get('region', {}).get('S','')

        region_2 = last_two_jobs[1].get('region', {}).get('S','')

        region = os.environ.get('REGION', 'unknown')

        if (region == region_1 or region == region_2):
            
            # If your table uses a composite key, carry the round as well
            round_value = int(last_two_jobs[0].get('round', {}).get('N','0'))

            # Perform federated averaging
            aggregated_model_uri, local_aggregated_model_uri = federated_average_neural_networks([model_1, model_2])
            
            # Update DynamoDB with aggregated model
            update_aggregation_status(dynamodb_client=dynamodb, table_name=tableName, aggregated_model_uri=aggregated_model_uri, local_aggregated_model_uri=local_aggregated_model_uri, region=region, round_value=round_value )

            # Trigger the orchestrator to train the updated Model

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Neural network models aggregated successfully',
                    'source_models': [model_1, model_2],
                    'aggregated_model': ""
                })
            }

        else:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Region bypassed',
                    'source_models': [model_1, model_2],
                    'aggregated_model': aggregated_model_uri
                })
            }

def federated_average_neural_networks(model_uris):
    """
    Download models, average their weights, and save aggregated model
    """
    try:
        print("Starting federated averaging...")
        temp_dir = tempfile.mkdtemp()
        
        # Download and extract all models
        model_weights = []
        model_metadata = None
        
        for i, uri in enumerate(model_uris):
            print(f"Processing model {i+1}: {uri}")
            
            # Parse S3 URI
            bucket, key = parse_s3_uri(uri)
            
            # Download model
            model_dir = download_and_extract_model(bucket, key, temp_dir, f"model_{i}")
            
            # Load weights
            weights_path = os.path.join(model_dir, 'model.pth')
            if os.path.exists(weights_path):
                weights = torch.load(weights_path, map_location='cpu')
                model_weights.append(weights)
                print(f"Loaded weights from model {i+1}")
            else:
                print(f"Warning: No weights found for model {i+1}")
                continue
            
            # Load metadata from first model
            if model_metadata is None:
                metadata_path = os.path.join(model_dir, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        model_metadata = json.load(f)
        
        if len(model_weights) < 2:
            raise Exception("Not enough model weights found for aggregation")
        
        # Perform federated averaging
        print(f"Averaging {len(model_weights)} sets of weights...")
        averaged_weights = average_model_weights(model_weights)
        
        # Create aggregated model
        aggregated_model_uri, local_aggregated_model_uri = create_aggregated_model(
            averaged_weights, model_metadata, temp_dir
        )
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return (aggregated_model_uri, local_aggregated_model_uri)
        
    except Exception as e:
        raise Exception(f"Federated averaging failed: {str(e)}")

def parse_s3_uri(s3_uri):
    """Parse S3 URI into bucket and key"""
    path = s3_uri.replace('s3://', '')
    parts = path.split('/', 1)
    return parts[0], parts[1] if len(parts) > 1 else ''

def download_and_extract_model(bucket, key, temp_dir, model_name):
    """Download and extract model tar.gz from S3"""
    try:
        # Download tar.gz
        tar_path = os.path.join(temp_dir, f"{model_name}.tar.gz")
        s3.download_file(bucket, key, tar_path)
        
        # Extract
        extract_dir = os.path.join(temp_dir, model_name)
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        
        return extract_dir
        
    except Exception as e:
        raise Exception(f"Failed to download model from s3://{bucket}/{key}: {str(e)}")

def average_model_weights(model_weights_list):
    """
    Average neural network weights using FedAvg algorithm
    """
    try:
        print("Performing federated averaging...")
        
        # Initialize averaged weights with zeros
        num_models = len(model_weights_list)
        averaged_weights = {}
        
        # Get the structure from the first model
        first_model_weights = model_weights_list[0]


        
        # Average each parameter
        for param_name in first_model_weights.keys():
            print(f"Averaging parameter: {param_name}")
            
            # Collect this parameter from all models
            param_tensors = []
            for model_weights in model_weights_list:
                if param_name in model_weights:
                    param_tensors.append(model_weights[param_name])
                else:
                    print(f"Warning: Parameter {param_name} not found in one of the models")
            
            if param_tensors:
                # Check if tensors are floating point, convert if needed
                first_tensor = param_tensors[0]
                
                if first_tensor.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.long]:
                    # For integer tensors (like indices), take the first model's values
                    # since averaging indices doesn't make sense
                    averaged_param = first_tensor.clone()
                    print(f"Using first model's values for integer parameter: {param_name}, dtype: {first_tensor.dtype}")
                else:
                    # Sum all tensors (convert to float only if they're integer types that should be averaged)
                    param_sum = None
                    for i, tensor in enumerate(param_tensors):
                        # Only convert integer types to float to preserve precision in division
                        if tensor.dtype in [torch.int8, torch.int16, torch.int32]:
                            tensor = tensor.float()
                        
                        if param_sum is None:
                            param_sum = tensor.clone()
                        else:
                            param_sum = param_sum + tensor
                    
                    # Divide by number of models to get average
                    averaged_param = param_sum / len(param_tensors)
                    print(f"Averaged {param_name}, shape: {averaged_param.shape}, dtype: {averaged_param.dtype}")
                
                averaged_weights[param_name] = averaged_param
        
        print(f"Successfully averaged weights from {num_models} models")
        return averaged_weights
        
    except Exception as e:
        raise Exception(f"Weight averaging failed: {str(e)}")

def create_aggregated_model(averaged_weights, metadata, temp_dir):
    """
    Create and save aggregated model to S3
    """
    try:
        print("Creating aggregated model...")
        
        # Create model directory
        model_dir = os.path.join(temp_dir, 'aggregated_model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save averaged weights
        weights_path = os.path.join(model_dir, 'model.pth')
        torch.save(averaged_weights, weights_path)
        
        # Update metadata
        if metadata:
            metadata['aggregation_timestamp'] = datetime.now().isoformat()
            metadata['aggregation_method'] = 'federated_averaging'
            metadata['num_models_aggregated'] = 2  # Update based on actual number
        else:
            metadata = {
                'framework': 'pytorch',
                'model_type': 'neural_network_aggregated',
                'aggregation_timestamp': datetime.now().isoformat(),
                'aggregation_method': 'federated_averaging'
            }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create tar.gz
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        tar_path = os.path.join(temp_dir, 'aggregated_model.tar.gz')
        
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(model_dir, arcname='.')
        
        # Upload to Global S3
        central_bucket = os.environ.get('MODEL_BUCKET', 'fl-modelbucket')
        aggregated_key = f'aggregated-models/federated-avg-{timestamp}/model.tar.gz'
        
        
        s3.upload_file(tar_path, central_bucket, aggregated_key)
        print(f"Uploaded aggregated model to s3://{central_bucket}/{aggregated_key}")

        # Upload to Local S3
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        local_bucket = os.environ.get('LOCAL_BUCKET')
        region = os.environ.get('REGION')
        time = datetime.now().strftime("%H%M%S")
        local_aggregated_key = f'models/{date_str}/federated-training-{region}-{time}/model.tar.gz'

        # central_bucket = os.environ.get('MODEL_BUCKET', 'fl-modelbucket')
        # aggregated_key = f'aggregated-models/federated-avg-{timestamp}/model.tar.gz'
        
        s3.upload_file(tar_path, local_bucket, local_aggregated_key)
        print(f"Uploaded local model to s3://{local_bucket}/{local_aggregated_key}")
        
        return (f's3://{central_bucket}/{aggregated_key}', f's3://{local_bucket}/{local_aggregated_key}')
        
    except Exception as e:
        raise Exception(f"Failed to create aggregated model: {str(e)}")

def update_aggregation_status(dynamodb_client, table_name, aggregated_model_uri, local_aggregated_model_uri, region, round_value):
    """
    Update DynamoDB with aggregation results
    """
    try:
        
        dynamodb_client.put_item(TableName=table_name, 
            Item= {
                "region":{'S': str(region)},        # String type
                "round":  {'N': str(round_value)},   # Number type (as string)
                "status": {'S': 'aggregated'},   # String type
                "model_s3_uri": {'S': str(local_aggregated_model_uri)},
                "central_bucket_uri": {'S': str(aggregated_model_uri)},
                "latest_uri": {'S': str(aggregated_model_uri)},
                "timestamp": {"S": _iso(datetime.now())},
                "created_at": {"S": _iso(datetime.now())},
                "training_start_time": {"S": _iso(datetime.now())},
                "training_end_time": {"S": _iso(datetime.now())},
                "model_type": {"S": "federated-learning"}
            }
        )

        print("Updated DynamoDB with aggregation results")
        
    except Exception as e:
        print(f"Failed to update DynamoDB: {str(e)}")
        # Don't fail the whole process if DynamoDB update fails