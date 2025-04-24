import os
import json
import yaml
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def save_model_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save model metrics to a JSON file.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing model metrics
        output_path (str): Path to save the metrics file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: List[str], output_path: str) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        labels (List[str]): Label names
        output_path (str): Path to save the plot
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                 labels: List[str], output_path: str) -> None:
    """
    Generate and save classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        labels (List[str]): Label names
        output_path (str): Path to save the report
    """
    try:
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report_df.to_csv(output_path)
        logger.info(f"Classification report saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        raise

def create_experiment_directory(base_path: str, experiment_name: str) -> str:
    """
    Create a directory for experiment outputs.
    
    Args:
        base_path (str): Base directory path
        experiment_name (str): Name of the experiment
        
    Returns:
        str: Path to the experiment directory
    """
    try:
        experiment_dir = os.path.join(base_path, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        logger.info(f"Created experiment directory: {experiment_dir}")
        return experiment_dir
    except Exception as e:
        logger.error(f"Error creating experiment directory: {e}")
        raise 