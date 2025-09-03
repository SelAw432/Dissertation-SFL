import json
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import boto3
import tarfile
import tempfile


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

s3 = boto3.client("s3")


class NeuralNetworkClassifier(nn.Module):
    """
    Simple neural network for classification with 80 features
    """
    def __init__(self, input_size=2, hidden_layers=None, num_classes=2, dropout_rate=0.2):
        super(NeuralNetworkClassifier, self).__init__()
       
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
       
        layers = []
        prev_size = input_size
       
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
       
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
       
        self.network = nn.Sequential(*layers)
       
    def forward(self, x):
        return self.network(x)


def download_and_extract_model(bucket_name, model_key, extract_path):
    """
    Download model.tar.gz from S3 and extract it
    """
    try:
        logger.info(f"Downloading model from s3://{bucket_name}/{model_key}")
        
        # Download the model tar file
        tar_path = os.path.join(extract_path, 'model.tar.gz')
        s3.download_file(bucket_name, model_key, tar_path)
        
        # Extract the tar file
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_path)
        
        # Remove the tar file to save space
        os.remove(tar_path)
        
        logger.info(f"Model extracted to {extract_path}")
        return extract_path
        
    except Exception as e:
        logger.error(f"Error downloading/extracting model: {str(e)}")
        raise


def find_latest_model(region, table_name, dynamodb_region):
    """
    Get the latest model info from DynamoDB for the specified region
    """
    try:
        logger.info(f"Getting model for region '{region}' from DynamoDB table: {table_name}")
        dynamodb = boto3.client("dynamodb", region_name='eu-west-1')
        # table = dynamodb.Table(table_name)
        
        # Get the item directly by region key
        response = dynamodb.get_item(
            TableName=table_name,
            Key={'region': {'S': region}}
        )
        # response = table.get_item(
        #     Key={'region': region}
        # )
        
        if 'Item' not in response:
            raise Exception(f"No model found for region '{region}' in DynamoDB table")

        latest_item = response['Item']

        # Check if model is in completed status
        status = latest_item.get('status', {}).get('S','')
        if status != 'completed':
            logger.warning(f"Model status is '{status}', not 'completed'")
        
        # Extract model path - prefer model_s3_uri over latest_uri
        model_uri = latest_item.get('model_s3_uri', {}).get('S','')

        if not model_uri:
            raise Exception("No model URI found in DynamoDB record")
        
        # Parse S3 URI to get bucket and key
        if model_uri.startswith('s3://'):
            uri_parts = model_uri[5:].split('/', 1)
            bucket_name = uri_parts[0]
            model_key = uri_parts[1] if len(uri_parts) > 1 else ''
        else:
            raise Exception(f"Invalid S3 URI format: {model_uri}")
        
        # Extract other values safely
        round_num = latest_item.get('round', {}).get('N', '0')
        region_value = latest_item.get('region', {}).get('S', region)

        logger.info(f"Latest model found: {model_uri}")
        logger.info(f"  Round: {round_num}")
        logger.info(f"  Status: {status}")
        logger.info(f"  Bucket: {bucket_name}")
        logger.info(f"  Key: {model_key}")
        
        result = {
            'round': round_num,
            'status': status,
            'region': region_value,
            'model_s3_uri': model_uri
        }
        return bucket_name, model_key, result
        
    except Exception as e:
        logger.error(f"Error querying DynamoDB: {str(e)}")
        raise


def load_model_components(model_dir, device):
    """
    Load model, and configuration
    """
    try:
        # Load configuration
        config_path = os.path.join(model_dir, 'model_config.json')
        if not os.path.exists(config_path):
            raise Exception(f"Model config not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
       
        # Initialize model
        model = NeuralNetworkClassifier(
            input_size=config['input_size'],
            hidden_layers=config['hidden_layers'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        ).to(device)
       
        # Load model weights
        model_path = os.path.join(model_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise Exception(f"Model weights not found at {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info("Model components loaded successfully")
        return model, config
        
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise


def preprocess_input_data(input_data, expected_features):
    """
    Preprocess and validate input data
    """
    try:
        # Convert input to numpy array
        if isinstance(input_data, list):
            input_array = np.array(input_data)
        elif isinstance(input_data, dict):
            # If input is a dictionary, extract values in order
            input_array = np.array(list(input_data.values()))
        else:
            input_array = np.array(input_data)
        
        # Ensure 2D array (batch dimension)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)
        
        # Validate input dimensions
        if input_array.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {input_array.shape[1]}")
        
        logger.info(f"Input data preprocessed: shape {input_array.shape}")
        return input_array
        
    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise


def handler(event, context):
    """
    Main Lambda handler for inference
    """
    try:
        logger.info("Starting inference Lambda function")
        
        # Get environment variables
        table_name = os.environ.get("MODEL_REGISTRY")
        region = os.environ.get("MODEL_REGION")
        dynamodb_region = os.environ.get("MODEL_REGISTRY_REGION")
        
        if not table_name:
            raise Exception("MODEL_REGISTRY environment variable not set")
        
        device = torch.device("cpu")
        
        # Extract input data from event
        logger.info("Extracting input data from event")
        input_data = event.get("data", event.get("body", {}))
        
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
        
        if not input_data:
            raise ValueError("No input data provided")
        
        # Create temporary directory for model files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get latest model info from DynamoDB
            logger.info("Getting latest model from DynamoDB")
            bucket_name, model_key, model_info = find_latest_model(region, table_name, dynamodb_region)
            
            logger.info("Downloading and extracting model")
            model_dir = download_and_extract_model(bucket_name, model_key, temp_dir)
            
            # Load model components
            logger.info("Loading model components")
            model, config = load_model_components(model_dir, device)
            
            # Preprocess input data
            logger.info("Preprocessing input data")
            input_scaled = preprocess_input_data(input_data, config['input_size'])
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_scaled).to(device)
            
            # Make predictions
            logger.info("Making predictions")
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
            
            # Prepare response
            response = {
                'statusCode': 200,
                'body': json.dumps({
                    'predictions': predictions.cpu().numpy().tolist(),
                    'probabilities': probabilities.cpu().numpy().tolist()
                })
            }
            
            logger.info("Inference completed successfully")
            logger.info(f"Used model from round {model_info.get('round')} in region {model_info.get('region')}")
            return response
            
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Inference failed'
            })
        }