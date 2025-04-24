import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Class for creating various visualizations of the dataset and results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")

    def plot_class_distribution(self, labels: List[int], class_names: List[str]):
        """
        Plot the distribution of classes in the dataset.
        
        Args:
            labels (List[int]): List of class labels
            class_names (List[str]): Names of the classes
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=labels)
        plt.title('Distribution of Sentiment Classes')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(range(len(class_names)), class_names)
        
        # Add count labels on top of bars
        for i in range(len(class_names)):
            count = (labels == i).sum()
            plt.text(i, count, str(count), ha='center', va='bottom')
        
        plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'))
        plt.close()
        logger.info("Class distribution plot saved")

    def create_wordcloud(self, texts: List[str], sentiment: str):
        """
        Create a word cloud for texts of a specific sentiment.
        
        Args:
            texts (List[str]): List of text samples
            sentiment (str): Sentiment class name
        """
        # Combine all texts
        text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment} Sentiment')
        
        plt.savefig(os.path.join(self.output_dir, f'wordcloud_{sentiment.lower()}.png'))
        plt.close()
        logger.info(f"Word cloud for {sentiment} sentiment saved")

    def plot_training_history(self, history: Dict[str, List[float]]):
        """
        Plot training history (loss and accuracy).
        
        Args:
            history (Dict[str, List[float]]): Dictionary containing training metrics
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()
        logger.info("Training history plot saved")

    def plot_confusion_matrix_heatmap(self, cm: np.ndarray, class_names: List[str]):
        """
        Create a heatmap of the confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            class_names (List[str]): Names of the classes
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix_heatmap.png'))
        plt.close()
        logger.info("Confusion matrix heatmap saved")

    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       class_names: List[str]):
        """
        Plot ROC curves for each class.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            class_names (List[str]): Names of the classes
        """
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr,
                tpr,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join(self.output_dir, 'roc_curves.png'))
        plt.close()
        logger.info("ROC curves plot saved")

    def plot_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: List[str], top_n: int = 20):
        """
        Plot feature importance scores.
        
        Args:
            feature_importance (np.ndarray): Importance scores
            feature_names (List[str]): Names of the features
            top_n (int): Number of top features to plot
        """
        # Sort features by importance
        indices = np.argsort(feature_importance)[-top_n:]
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(top_n), feature_importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Features by Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()
        logger.info("Feature importance plot saved")

def create_visualization_report(data: Dict[str, Any], output_dir: str):
    """
    Create a comprehensive visualization report.
    
    Args:
        data (Dict[str, Any]): Dictionary containing all necessary data
        output_dir (str): Directory to save visualizations
    """
    visualizer = DataVisualizer(output_dir)
    
    # Plot class distribution
    visualizer.plot_class_distribution(
        data['labels'],
        data['class_names']
    )
    
    # Create word clouds for each sentiment
    for sentiment, texts in data['texts_by_sentiment'].items():
        visualizer.create_wordcloud(texts, sentiment)
    
    # Plot training history
    if 'training_history' in data:
        visualizer.plot_training_history(data['training_history'])
    
    # Plot confusion matrix
    if 'confusion_matrix' in data:
        visualizer.plot_confusion_matrix_heatmap(
            data['confusion_matrix'],
            data['class_names']
        )
    
    # Plot ROC curves
    if 'y_true' in data and 'y_pred_proba' in data:
        visualizer.plot_roc_curves(
            data['y_true'],
            data['y_pred_proba'],
            data['class_names']
        )
    
    # Plot feature importance if available
    if 'feature_importance' in data and 'feature_names' in data:
        visualizer.plot_feature_importance(
            data['feature_importance'],
            data['feature_names']
        )
    
    logger.info("Visualization report created successfully") 