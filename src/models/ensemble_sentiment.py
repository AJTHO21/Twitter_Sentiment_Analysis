import torch
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import copy
import sys
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwitterDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: RobertaTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert labels to integers
        self.label_map = {
            'Negative': 0,
            'Neutral': 1,
            'Positive': 2
        }
        self.int_labels = [self.label_map[label] for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.int_labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EnsembleSentimentAnalyzer:
    def __init__(self, output_dir: str = 'data/analysis/ensemble'):
        """Initialize the ensemble sentiment analyzer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer
        self.model_name = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        # Load the three trained models
        self.models = {}
        self.load_models()
        
        # Label mapping
        self.label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
    def load_models(self):
        """Load the three trained models."""
        try:
            # Define paths to the saved models
            model_paths = {
                'Positive': 'data/analysis/fine_tuned/checkpoints/positive_model',
                'Negative': 'data/analysis/fine_tuned/checkpoints/negative_model',
                'Neutral': 'data/analysis/fine_tuned/checkpoints/neutral_model'
            }
            
            # Load each model
            for sentiment, path in model_paths.items():
                if os.path.exists(path):
                    logger.info(f"Loading {sentiment} model from {path}")
                    model = RobertaForSequenceClassification.from_pretrained(path)
                    self.models[sentiment] = model
                else:
                    logger.warning(f"Model path {path} does not exist. Skipping {sentiment} model.")
            
            if not self.models:
                raise ValueError("No models were loaded. Check if model paths are correct.")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(self, texts: List[str]) -> List[str]:
        """Make predictions using the ensemble of models."""
        try:
            # Create dataset
            dataset = TwitterDataset(texts, ['Neutral'] * len(texts), self.tokenizer)
            
            # Get predictions from each model
            all_predictions = []
            for sentiment, model in self.models.items():
                logger.info(f"Getting predictions from {sentiment} model")
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=os.path.join(self.output_dir, 'temp'),
                        per_device_eval_batch_size=16
                    ),
                    eval_dataset=dataset
                )
                
                # Get predictions
                predictions = trainer.predict(dataset)
                all_predictions.append(predictions.predictions)
            
            # Average predictions
            ensemble_predictions = np.mean(all_predictions, axis=0)
            
            # Get final predictions
            y_pred = np.argmax(ensemble_predictions, axis=1)
            y_pred_labels = [self.label_map[pred] for pred in y_pred]
            
            return y_pred_labels
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def evaluate(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """Evaluate the ensemble on a test set."""
        try:
            # Filter out irrelevant tweets
            df = df[df['sentiment'] != 'Irrelevant'].copy()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['text'],
                df['sentiment'],
                test_size=test_size,
                random_state=42,
                stratify=df['sentiment']
            )
            
            # Make predictions
            logger.info("Making predictions on test set")
            y_pred = self.predict(X_test.tolist())
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            # Store results
            results = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive']
            )
            plt.title('Confusion Matrix - Ensemble RoBERTa')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'confusion_matrix_ensemble.png'))
            plt.close()
            
            # Generate report
            self.generate_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            raise
    
    def generate_report(self, results: Dict):
        """Generate a detailed report of the ensemble's performance."""
        try:
            report = []
            report.append("# Ensemble Sentiment Analysis Results")
            report.append("\n## Model: Ensemble of Fine-tuned RoBERTa Models")
            
            # Add accuracy
            report.append(f"\n## Accuracy: {results['accuracy']:.3f}")
            
            # Add classification report
            report.append("\n## Classification Report")
            report.append("```")
            report.append(classification_report(
                results['true_labels'],
                results['predictions'],
                labels=['Negative', 'Neutral', 'Positive']
            ))
            report.append("```")
            
            # Save report
            report_path = os.path.join(self.output_dir, 'ensemble_report.md')
            logger.info(f"Saving report to {report_path}")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

def main():
    print("Starting ensemble sentiment analysis...")
    try:
        # Initialize ensemble analyzer
        print("Step 1: Initializing ensemble analyzer...")
        logger.info("Initializing EnsembleSentimentAnalyzer...")
        analyzer = EnsembleSentimentAnalyzer()
        print("Ensemble analyzer initialized successfully")
        
        # Load data
        print("Step 2: Loading data...")
        logger.info("Loading data...")
        data_path = 'data/processed/cleaned_twitter_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded {len(df)} tweets successfully")
            logger.info(f"Successfully loaded {len(df)} tweets from dataset")
            logger.info(f"Columns in dataset: {df.columns.tolist()}")
            logger.info(f"Sample of sentiment distribution:\n{df['sentiment'].value_counts()}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
            raise
        
        # Evaluate ensemble
        print("Step 3: Evaluating ensemble...")
        logger.info("Evaluating ensemble...")
        try:
            results = analyzer.evaluate(df)
            print("Ensemble evaluation completed")
            logger.info("Ensemble evaluation completed successfully")
            logger.info(f"Evaluation results: {results}")
        except Exception as e:
            print(f"Error during ensemble evaluation: {str(e)}")
            logger.error(f"Error during ensemble evaluation: {str(e)}")
            raise
        
        print("Analysis complete! Check the output directory for results.")
        logger.info("Analysis complete! Check the output directory for results.")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 