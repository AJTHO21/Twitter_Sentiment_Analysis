import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    RobertaForSequenceClassification,
    XLNetForSequenceClassification,
    GPT2ForSequenceClassification
)
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
import logging
import traceback
import argparse
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    def __init__(self, output_dir: str = 'data/analysis/enhanced_models', sample_size: float = 0.001):
        """Initialize the enhanced sentiment analyzer with multiple models."""
        self.output_dir = output_dir
        self.sample_size = sample_size
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'roberta': {
                'name': 'roberta-base',
                'model': None,
                'tokenizer': None,
                'pipeline': None
            },
            'xlnet': {
                'name': 'xlnet-base-cased',
                'model': None,
                'tokenizer': None,
                'pipeline': None
            },
            'bert-large': {
                'name': 'bert-large-uncased',
                'model': None,
                'tokenizer': None,
                'pipeline': None
            },
            'gpt2': {
                'name': 'gpt2',
                'model': None,
                'tokenizer': None,
                'pipeline': None
            }
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def setup_model(self, model_name: str):
        """Set up a specific pre-trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            logger.info(f"Loading {model_name}...")
            model_info = self.models[model_name]
            
            # Add progress bar for tokenizer loading
            with tqdm(total=2, desc="Loading model components") as pbar:
                model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
                pbar.update(1)
                
                model_info['model'] = AutoModelForSequenceClassification.from_pretrained(
                    model_info['name'],
                    num_labels=4  # Negative, Neutral, Positive, Irrelevant
                ).to(self.device)
                pbar.update(1)
            
            # Add progress bar for pipeline setup
            with tqdm(total=1, desc="Setting up pipeline") as pbar:
                model_info['pipeline'] = pipeline(
                    "sentiment-analysis",
                    model=model_info['model'],
                    tokenizer=model_info['tokenizer'],
                    device=0 if torch.cuda.is_available() else -1
                )
                pbar.update(1)
            
            logger.info(f"{model_name} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def predict_sentiment(self, text: str, model_name: str) -> Tuple[str, float]:
        """Predict sentiment using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.models[model_name]
        result = model_info['pipeline'](text)[0]
        
        # Map model output to our sentiment categories
        label = result['label']
        score = result['score']
        
        # Convert model-specific labels to our categories
        if 'POSITIVE' in label.upper():
            return 'Positive', score
        elif 'NEGATIVE' in label.upper():
            return 'Negative', score
        elif 'NEUTRAL' in label.upper():
            return 'Neutral', score
        else:
            return 'Irrelevant', score

    def evaluate_model(self, df: pd.DataFrame, model_name: str) -> Dict:
        """Evaluate a specific model on the dataset."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            logger.info(f"Evaluating {model_name}...")
            
            # Get predictions with progress bar
            predictions = []
            confidences = []
            
            # Create progress bar for predictions
            pbar = tqdm(total=len(df), desc="Processing tweets")
            for text in df['text']:
                pred, conf = self.predict_sentiment(text, model_name)
                predictions.append(pred)
                confidences.append(conf)
                pbar.update(1)
            pbar.close()
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            accuracy = np.mean(predictions == df['sentiment'])
            report = classification_report(df['sentiment'], predictions, output_dict=True)
            cm = confusion_matrix(
                df['sentiment'],
                predictions,
                labels=['Negative', 'Neutral', 'Positive', 'Irrelevant']
            )
            
            # Store results
            results = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm,
                'confidences': confidences,
                'predictions': predictions
            }
            
            # Create confusion matrix plot with progress bar
            logger.info("Generating confusion matrix...")
            with tqdm(total=1, desc="Creating visualization") as pbar:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'],
                    yticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant']
                )
                plt.title(f'Confusion Matrix - {model_name}')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png'))
                plt.close()
                pbar.update(1)
            
            logger.info(f"{model_name} evaluation completed successfully")
            
            # Generate report for this model
            self.generate_model_report(model_name, results, df)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def generate_model_report(self, model_name: str, results: Dict, df: pd.DataFrame):
        """Generate a report for a specific model."""
        try:
            logger.info("Generating report...")
            report = []
            report.append(f"# {model_name} Sentiment Analysis Results")
            report.append("\n## Performance Summary")
            
            # Add accuracy
            report.append(f"\n### Accuracy: {results['accuracy']:.3f}")
            
            # Add classification report
            report.append("\n### Classification Report")
            report.append("```")
            report.append(classification_report(
                df['sentiment'],
                results['predictions'],
                labels=['Negative', 'Neutral', 'Positive', 'Irrelevant']
            ))
            report.append("```")
            
            # Save report
            report_path = os.path.join(self.output_dir, f'{model_name}_report.md')
            logger.info(f"Saving {model_name} report to {report_path}")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            
        except Exception as e:
            logger.error(f"Error generating report for {model_name}: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run sentiment analysis with a specific model')
    parser.add_argument('--model', type=str, choices=['roberta', 'xlnet', 'bert-large', 'gpt2'], 
                        help='Model to use for sentiment analysis')
    parser.add_argument('--sample-size', type=float, default=0.001, 
                        help='Sample size as a fraction of the dataset (default: 0.001)')
    args = parser.parse_args()
    
    if not args.model:
        logger.error("Please specify a model using --model")
        return
    
    try:
        # Initialize analyzer with specified sample size
        analyzer = EnhancedSentimentAnalyzer(sample_size=args.sample_size)
        
        # Load and prepare data
        logger.info("Loading data...")
        df = pd.read_csv('data/processed/cleaned_twitter_data.csv')
        
        # Take a stratified sample
        logger.info(f"Taking a {analyzer.sample_size*100}% stratified sample...")
        _, df_sample = train_test_split(
            df,
            train_size=analyzer.sample_size,
            stratify=df['sentiment'],
            random_state=42
        )
        logger.info(f"Sample size: {len(df_sample)} tweets")
        
        # Set up the specified model
        if not analyzer.setup_model(args.model):
            logger.error(f"Failed to set up {args.model}. Exiting.")
            return
        
        # Evaluate the model
        results = analyzer.evaluate_model(df_sample, args.model)
        
        if results:
            logger.info(f"Analysis complete! Check the output directory for {args.model} results.")
        else:
            logger.error(f"Analysis failed for {args.model}.")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 