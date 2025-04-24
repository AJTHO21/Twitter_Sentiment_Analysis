import os
import logging
import yaml
from typing import Dict, Any
import numpy as np
import torch
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import load_data
from src.models.sentiment_model import SentimentAnalyzer
from src.utils.helpers import (
    save_model_metrics,
    plot_confusion_matrix,
    generate_classification_report,
    create_experiment_directory
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_experiment(config: Dict[str, Any]) -> str:
    """
    Set up the experiment directory and logging.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Path to the experiment directory
    """
    # Create experiment directory
    experiment_dir = create_experiment_directory(
        config['output']['base_dir'],
        'sentiment_analysis'
    )
    
    # Set up logging file
    log_file = os.path.join(experiment_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(config['logging']['format'])
    )
    logger.addHandler(file_handler)
    
    return experiment_dir

def train_model(config: Dict[str, Any], experiment_dir: str):
    """
    Train the sentiment analysis model.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        experiment_dir (str): Path to the experiment directory
    """
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, y_train, X_val, y_val = load_data(
            config['data']['train_path'],
            config['data']['val_path']
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model = SentimentAnalyzer(config)
        model.setup_model()
        
        # Create dataloaders
        train_loader, val_loader = model.create_dataloaders(
            X_train, y_train, X_val, y_val
        )
        
        # Train model
        logger.info("Starting training...")
        model.train(train_loader, val_loader)
        
        # Evaluate model
        logger.info("Evaluating model...")
        model.model.load_state_dict(
            torch.load(os.path.join(config['output']['model_dir'], 'best_model.pt'))
        )
        
        # Generate predictions
        model.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                outputs = model.model(input_ids, attention_mask)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Save metrics and plots
        logger.info("Saving results...")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(all_labels),
            np.array(all_preds),
            ['Negative', 'Neutral', 'Positive'],
            os.path.join(experiment_dir, 'confusion_matrix.png')
        )
        
        # Generate classification report
        generate_classification_report(
            np.array(all_labels),
            np.array(all_preds),
            ['Negative', 'Neutral', 'Positive'],
            os.path.join(experiment_dir, 'classification_report.csv')
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

def main():
    """Main function to run the training process."""
    try:
        # Load configuration
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create necessary directories
        os.makedirs(config['output']['model_dir'], exist_ok=True)
        os.makedirs(config['output']['plots_dir'], exist_ok=True)
        os.makedirs(config['output']['reports_dir'], exist_ok=True)
        
        # Set up experiment
        experiment_dir = setup_experiment(config)
        
        # Train model
        train_model(config, experiment_dir)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 