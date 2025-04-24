import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TraditionalMLSentimentAnalyzer:
    def __init__(self, output_dir: str = 'data/analysis/traditional_ml', sample_size: float = 0.001):
        """Initialize the traditional ML sentiment analyzer."""
        self.output_dir = output_dir
        self.sample_size = sample_size
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)  # Include both single words and pairs
        )
        self.model = SVC(
            kernel='linear',
            probability=True,
            random_state=42
        )
        
        # Parameters for grid search
        self.param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'class_weight': ['balanced', None]
        }

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and sample the data."""
        logger.info(f"Taking a {self.sample_size*100}% stratified sample...")
        _, df_sample = train_test_split(
            df,
            train_size=self.sample_size,
            stratify=df['sentiment'],
            random_state=42
        )
        logger.info(f"Sample size: {len(df_sample)} tweets")
        return df_sample

    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train the model with hyperparameter tuning."""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['text'],
                df['sentiment'],
                test_size=0.2,
                random_state=42,
                stratify=df['sentiment']
            )
            
            # Vectorize text
            logger.info("Vectorizing text data...")
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
            
            # Grid search for hyperparameter tuning
            logger.info("Performing grid search for hyperparameter tuning...")
            grid_search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            with tqdm(total=1, desc="Training model") as pbar:
                grid_search.fit(X_train_tfidf, y_train)
                pbar.update(1)
            
            # Get best model
            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Make predictions
            logger.info("Making predictions...")
            y_pred = self.model.predict(X_test_tfidf)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=['Negative', 'Neutral', 'Positive', 'Irrelevant']
            )
            
            # Store results
            results = {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm,
                'best_params': grid_search.best_params_,
                'predictions': y_pred,
                'true_labels': y_test
            }
            
            # Create confusion matrix plot
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
                plt.title('Confusion Matrix - SVM with TF-IDF')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'confusion_matrix_svm.png'))
                plt.close()
                pbar.update(1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}")
            raise

    def generate_report(self, results: Dict):
        """Generate a detailed report of the model's performance."""
        try:
            report = []
            report.append("# Traditional ML Sentiment Analysis Results")
            report.append("\n## Model: SVM with TF-IDF")
            
            # Add best parameters
            report.append("\n## Best Parameters")
            report.append("```")
            for param, value in results['best_params'].items():
                report.append(f"{param}: {value}")
            report.append("```")
            
            # Add accuracy
            report.append(f"\n## Accuracy: {results['accuracy']:.3f}")
            
            # Add classification report
            report.append("\n## Classification Report")
            report.append("```")
            report.append(classification_report(
                results['true_labels'],
                results['predictions'],
                labels=['Negative', 'Neutral', 'Positive', 'Irrelevant']
            ))
            report.append("```")
            
            # Save report
            report_path = os.path.join(self.output_dir, 'svm_report.md')
            logger.info(f"Saving report to {report_path}")
            with open(report_path, 'w') as f:
                f.write('\n'.join(report))
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

def main():
    try:
        # Initialize analyzer
        analyzer = TraditionalMLSentimentAnalyzer()
        
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('data/processed/cleaned_twitter_data.csv')
        
        # Prepare data
        df_sample = analyzer.prepare_data(df)
        
        # Train model and get results
        results = analyzer.train_model(df_sample)
        
        # Generate report
        analyzer.generate_report(results)
        
        logger.info("Analysis complete! Check the output directory for results.")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 