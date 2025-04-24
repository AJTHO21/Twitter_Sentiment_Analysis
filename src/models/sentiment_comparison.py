import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from typing import Dict, List, Tuple, Callable
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove special characters but keep emoticons and basic punctuation
    text = re.sub(r'[^\w\s!?.,@#]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def vader_sentiment(text: str) -> str:
    """Get sentiment using VADER."""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def textblob_sentiment(text: str) -> str:
    """Get sentiment using TextBlob."""
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def distilbert_sentiment(text: str, classifier) -> str:
    """Get sentiment using DistilBERT."""
    result = classifier(text)[0]
    label = result['label']
    
    # Map DistilBERT labels to our categories
    if 'POSITIVE' in label:
        return 'Positive'
    elif 'NEGATIVE' in label:
        return 'Negative'
    else:
        return 'Neutral'

def evaluate_sentiment_analyzer(df: pd.DataFrame, analyzer_func: Callable, name: str, 
                              output_dir: str, classifier=None) -> Dict:
    """Evaluate a sentiment analyzer on the dataset."""
    print(f"Evaluating {name}...")
    
    # Apply the sentiment analyzer
    if classifier:
        df[f'{name}_sentiment'] = df['cleaned_text'].apply(lambda x: analyzer_func(x, classifier))
    else:
        df[f'{name}_sentiment'] = df['cleaned_text'].apply(analyzer_func)
    
    # Calculate accuracy
    accuracy = accuracy_score(df['sentiment'], df[f'{name}_sentiment'])
    
    # Generate classification report
    report = classification_report(df['sentiment'], df[f'{name}_sentiment'], output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(df['sentiment'], df[f'{name}_sentiment'], 
                         labels=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'],
                yticklabels=['Negative', 'Neutral', 'Positive', 'Irrelevant'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name.lower()}.png'))
    plt.close()
    
    return {
        'name': name,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def main():
    # Create output directory
    output_dir = 'data/analysis/sentiment_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read cleaned data
    print("Reading cleaned data...")
    df = pd.read_csv('data/processed/cleaned_twitter_data.csv')
    
    # Clean text data
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Initialize DistilBERT classifier
    print("Initializing DistilBERT classifier...")
    distilbert_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Evaluate different sentiment analyzers
    results = []
    
    # VADER
    vader_results = evaluate_sentiment_analyzer(df, vader_sentiment, 'VADER', output_dir)
    results.append(vader_results)
    
    # TextBlob
    textblob_results = evaluate_sentiment_analyzer(df, textblob_sentiment, 'TextBlob', output_dir)
    results.append(textblob_results)
    
    # DistilBERT
    distilbert_results = evaluate_sentiment_analyzer(df, distilbert_sentiment, 'DistilBERT', output_dir, distilbert_classifier)
    results.append(distilbert_results)
    
    # Plot accuracy comparison
    accuracies = [r['accuracy'] for r in results]
    names = [r['name'] for r in results]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=accuracies)
    plt.title('Sentiment Analysis Accuracy Comparison')
    plt.xlabel('Library')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Generate detailed report
    report = []
    report.append("# Sentiment Analysis Library Comparison")
    report.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append("\n## Overall Accuracy")
    for result in results:
        report.append(f"- {result['name']}: {result['accuracy']:.3f}")
    
    report.append("\n## Detailed Results")
    for result in results:
        report.append(f"\n### {result['name']}")
        report.append("\n#### Classification Report")
        report.append("```")
        report.append(classification_report(df['sentiment'], df[f"{result['name']}_sentiment"]))
        report.append("```")
    
    # Save report
    with open(os.path.join(output_dir, 'sentiment_comparison_report.md'), 'w') as f:
        f.write('\n'.join(report))
    
    print("\nComparison complete! Check the following files:")
    print(f"- Detailed report: {os.path.join(output_dir, 'sentiment_comparison_report.md')}")
    print("- Visualizations in the sentiment_comparison directory")
    
    # Return the best performing library
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest performing library: {best_result['name']} (Accuracy: {best_result['accuracy']:.3f})")

if __name__ == "__main__":
    main() 