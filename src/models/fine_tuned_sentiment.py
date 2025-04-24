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
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwitterDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: RobertaTokenizer, classifier_type: str, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classifier_type = classifier_type
        
        # Convert labels to integers for the specific binary classifier
        self.label_map = {
            'pos_neg': {'Positive': 1, 'Negative': 0},
            'pos_neu': {'Positive': 1, 'Neutral': 0},
            'neg_neu': {'Negative': 1, 'Neutral': 0}
        }[classifier_type]
        
        # Filter samples for this classifier
        valid_labels = set(self.label_map.keys())
        self.valid_indices = [i for i, label in enumerate(labels) if label in valid_labels]
        self.filtered_texts = [texts[i] for i in self.valid_indices]
        self.filtered_labels = [labels[i] for i in self.valid_indices]
        self.int_labels = [self.label_map[label] for label in self.filtered_labels]

    def __len__(self):
        return len(self.filtered_texts)

    def __getitem__(self, idx):
        text = str(self.filtered_texts[idx])
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

class UnifiedSentimentAnalyzer:
    def __init__(self, output_dir: str = 'data/analysis/fine_tuned', n_samples: int = 75):
        """Initialize the unified sentiment analyzer."""
        self.output_dir = output_dir
        self.n_samples = n_samples
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing model and tokenizer...")
        # Initialize model and tokenizer
        self.model_name = 'roberta-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        # Initialize model for positive vs negative classifier
        print("Setting up positive vs negative classifier...")
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            classifier_dropout=0.1
        )
        
        # Training arguments optimized for faster training
        print("Setting up training arguments...")
        self.training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, 'checkpoints'),
            num_train_epochs=10,  # Reduced epochs
            per_device_train_batch_size=16,  # Increased batch size
            per_device_eval_batch_size=16,
            weight_decay=0.1,
            learning_rate=3e-5,  # Slightly increased learning rate
            lr_scheduler_type="cosine_with_restarts",
            warmup_ratio=0.1,  # Reduced warmup
            warmup_steps=50,
            logging_dir=os.path.join(output_dir, 'logs'),
            save_strategy="epoch",
            max_grad_norm=0.5,
            optim="adamw_torch",
            fp16=True,
            gradient_checkpointing=True,
            disable_tqdm=False,
            report_to="none"
        )

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and sample the data for positive vs negative classification."""
        print(f"Taking {self.n_samples} samples per class...")
        
        # Filter for positive and negative tweets only
        df = df[df['sentiment'].isin(['Positive', 'Negative'])].copy()
        print(f"Initial dataset size after filtering: {len(df)}")
        
        # Get samples for each class
        pos_df = df[df['sentiment'] == 'Positive'].copy()
        neg_df = df[df['sentiment'] == 'Negative'].copy()
        
        print(f"Found {len(pos_df)} positive and {len(neg_df)} negative tweets")
        
        # Simplified feature engineering
        for temp_df in [pos_df, neg_df]:
            temp_df['text_length'] = temp_df['text'].str.len()
            temp_df['word_count'] = temp_df['text'].str.split().str.len()
        
        # Sample from each class
        n_samples = min(self.n_samples, len(pos_df), len(neg_df))
        pos_samples = pos_df.sample(n=n_samples, random_state=42)
        neg_samples = neg_df.sample(n=n_samples, random_state=42)
        
        # Combine samples
        df_sample = pd.concat([pos_samples, neg_samples])
        print(f"Final sample size: {len(df_sample)} tweets")
        print(f"Sample distribution:\n{df_sample['sentiment'].value_counts()}")
        
        return df_sample.reset_index(drop=True)

    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train the positive vs negative classifier."""
        try:
            print("Splitting data...")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['text'],
                df['sentiment'],
                test_size=0.2,
                random_state=42,
                stratify=df['sentiment']
            )
            
            print("Creating datasets...")
            # Create datasets
            train_dataset = TwitterDataset(X_train.tolist(), y_train.tolist(), self.tokenizer, 'pos_neg')
            eval_dataset = TwitterDataset(X_test.tolist(), y_test.tolist(), self.tokenizer, 'pos_neg')
            
            print(f"Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
            
            print("Starting training...")
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics
            )
            
            # Train the model
            trainer.train()
            print("Training completed!")
            
            print("Getting predictions...")
            # Get predictions
            predictions = trainer.predict(eval_dataset)
            
            # Convert predictions to labels
            y_pred = np.argmax(predictions.predictions, axis=1)
            label_map = {v: k for k, v in train_dataset.label_map.items()}
            y_pred_labels = [label_map[pred] for pred in y_pred]
            
            # Calculate metrics
            accuracy = np.mean([pred == true for pred, true in zip(y_pred_labels, y_test)])
            report = classification_report(y_test, y_pred_labels, output_dict=True)
            
            results = {
                'accuracy': accuracy,
                'report': report,
                'predictions': y_pred_labels,
                'true_labels': y_test.tolist()
            }
            
            print(f"Training complete! Accuracy: {accuracy:.3f}")
            return results
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Convert predictions back to labels
        label_map = {0: 'Negative', 1: 'Positive'}
        pred_labels = [label_map[pred] for pred in predictions]
        true_labels = [label_map[label] for label in labels]
        
        return {
            'accuracy': np.mean(pred_labels == true_labels)
        }

    def generate_report(self, results: Dict):
        """Generate a detailed report of the model's performance."""
        try:
            report = []
            report.append("# Binary Sentiment Analysis Results")
            report.append("\n## Model: Fine-tuned RoBERTa")
            report.append(f"\n## Accuracy: {results['accuracy']:.3f}")
            report.append("\n## Classification Report")
            report.append("```")
            report.append(classification_report(
                results['true_labels'],
                results['predictions']
            ))
            report.append("```")
            
            # Save report
            report_path = os.path.join(self.output_dir, 'binary_report.md')
            print(f"Saving report to {report_path}")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def main():
    try:
        print("Starting script execution...")
        
        # Create necessary directories
        print("Creating output directories...")
        os.makedirs('models/fine_tuned', exist_ok=True)
        os.makedirs('data/analysis/fine_tuned', exist_ok=True)
        
        print("Loading data from CSV...")
        try:
            df = pd.read_csv('data/processed/cleaned_twitter_data.csv')
            print(f"Successfully loaded {len(df)} rows from CSV")
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            raise
        
        print("Initializing UnifiedSentimentAnalyzer...")
        try:
            analyzer = UnifiedSentimentAnalyzer(n_samples=75)
            print("Analyzer initialized successfully")
        except Exception as e:
            print(f"Error initializing analyzer: {str(e)}")
            raise
        
        print("Preparing data...")
        try:
            prepared_data = analyzer.prepare_data(df)
            print(f"Data prepared successfully. Shape: {prepared_data.shape}")
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise
        
        print("Starting model training...")
        try:
            results = analyzer.train_model(prepared_data)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        
        print("Generating report...")
        try:
            analyzer.generate_report(results)
            print("Report generated successfully!")
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise
        
        print("Script completed successfully!")
        
    except Exception as e:
        print(f"\nFatal error in main: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 