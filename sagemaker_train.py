import argparse
import json
import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class NeuralNetworkClassifier(nn.Module):
    """
    Simple neural network for classification with 80 features
    """
    def __init__(self, input_size=80, hidden_layers=None, num_classes=2, dropout_rate=0.2):
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

def load_data(data_path, is_training=True):
    """
    Load and preprocess data from CSV file
    """
    logger.info(f"Loading data from {data_path}")
   
    # Find CSV files in the directory
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    if data_path is None:
        raise ValueError("data_path is None. Pass --train /path/to/data or set SM_CHANNEL_TRAINING.")
    if not os.path.isdir(data_path):
        raise ValueError(f"{data_path} is not a directory or doesn't exist.")
    csv_files = [f for f in os.listdir(data_path) if f.lower().endswith('.csv')]
    if not csv_files:
        raise ValueError(f"Error Message: No CSV files found in {data_path}")
   
    # Load the first CSV file found
    data_file = os.path.join(data_path, csv_files[0])
    df = pd.read_csv(data_file)
   
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
   
    # Assume the label column is named 'label' or is the last column
    if 'label' in df.columns:
        label_col = 'label'
    else:
        label_col = df.columns[-1]
        logger.info(f"Using column '{label_col}' as label")
   
    # Separate features and labels
    X = df.drop(label_col, axis=1).values
    y = df[label_col].values
   
    # Ensure labels are integers starting from 0
    unique_labels = np.unique(y)
    if not all(isinstance(label, (int, np.integer)) for label in unique_labels):
        # Convert string labels to integers
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])
        logger.info(f"Label mapping: {label_map}")
   
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Label shape: {y.shape}")
    logger.info(f"Unique labels: {np.unique(y)}")
   
    return X, y, len(unique_labels)

def train_model(args):
    """
    Main training function
    """
    logger.info("Starting training...")
   
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
   
    # Load training data
    X_train, y_train, num_classes = load_data(args.train)
   
    # Load validation data if available
    X_val, y_val = None, None
    if os.path.exists(args.validation):
        try:
            X_val, y_val, _ = load_data(args.validation)
            logger.info("Validation data loaded")
        except:
            logger.warning("Could not load validation data")
   
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
   
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
   
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
   
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
   
    val_loader = None
    if X_val is not None:
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
   
    # Parse hidden layers
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
   
    # Initialize model
    model = NeuralNetworkClassifier(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        dropout_rate=args.dropout
    ).to(device)
   
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
   
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
   
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
       
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
           
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
           
            epoch_loss += loss.item()
       
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
       
        # Validation phase
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_predictions = []
            val_targets = []
           
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                   
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
           
            val_acc = accuracy_score(val_targets, val_predictions)
            val_accuracies.append(val_acc)
           
            if val_acc > best_val_acc:
                best_val_acc = val_acc
       
        scheduler.step()
       
        # Log progress
        if (epoch + 1) % 10 == 0:
            if val_loader is not None:
                logger.info(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                logger.info(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')
   
    # Final evaluation on training set
    model.eval()
    train_predictions = []
    train_targets = []
   
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
           
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
   
    train_acc = accuracy_score(train_targets, train_predictions)
    logger.info(f'Final Training Accuracy: {train_acc:.4f}')
   
    if val_loader is not None:
        logger.info(f'Best Validation Accuracy: {best_val_acc:.4f}')
        logger.info("Validation Classification Report:")
        logger.info(classification_report(val_targets, val_predictions))
   
    # Save model and scaler
    model_path = os.path.join(args.model_dir, 'model.pth')
    scaler_path = os.path.join(args.model_dir, 'scaler.pkl')
   
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
   
    # Save model configuration
    model_config = {
        'input_size': X_train.shape[1],
        'hidden_layers': hidden_layers,
        'num_classes': num_classes,
        'dropout_rate': args.dropout
    }
   
    config_path = os.path.join(args.model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f)
   
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Config saved to {config_path}")

def model_fn(model_dir):
    """
    Load model for inference (SageMaker requirement)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Load model configuration
    config_path = os.path.join(model_dir, 'model_config.json')
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
   
    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
   
    return {'model': model, 'scaler': scaler, 'device': device}

def input_fn(request_body, request_content_type='application/json'):
    """
    Parse input data for inference (SageMaker requirement)
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['instances'])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """
    Make predictions (SageMaker requirement)
    """
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    device = model_artifacts['device']
   
    # Scale input data
    input_scaled = scaler.transform(input_data)
   
    # Convert to tensor
    input_tensor = torch.FloatTensor(input_scaled).to(device)
   
    # Make predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
   
    return {
        'predictions': predictions.cpu().numpy().tolist(),
        'probabilities': probabilities.cpu().numpy().tolist()
    }

def output_fn(prediction, accept='application/json'):
    """
    Format output (SageMaker requirement)
    """
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/train/'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', ''))
   
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-layers', type=str, default='128,64,32')
    parser.add_argument('--dropout', type=float, default=0.2)
   
    args = parser.parse_args()
    
    train_model(args)
