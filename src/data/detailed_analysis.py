import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from datetime import datetime

def analyze_sentiment_patterns(df: pd.DataFrame, output_dir: str):
    """Analyze detailed sentiment patterns and relationships."""
    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Calculate VADER scores for each text
    df['vader_scores'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    # Plot VADER score distribution by sentiment
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sentiment', y='vader_scores', data=df)
    plt.title('VADER Sentiment Score Distribution by Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vader_scores_by_sentiment.png'))
    plt.close()
    
    # Analyze sentiment agreement/disagreement
    df['vader_label'] = df['vader_scores'].apply(lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral')
    agreement = (df['sentiment'] == df['vader_label']).mean() * 100
    
    # Plot agreement heatmap
    agreement_matrix = pd.crosstab(df['sentiment'], df['vader_label'], normalize='index') * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Sentiment Label vs VADER Score Agreement (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_agreement_heatmap.png'))
    plt.close()
    
    return agreement

def analyze_topic_sentiment_relationships(df: pd.DataFrame, output_dir: str):
    """Analyze relationships between topics and sentiments."""
    # Calculate sentiment ratios for each topic
    topic_sentiment = pd.crosstab(df['topic'], df['sentiment'], normalize='index')
    
    # Calculate topic-specific metrics
    topic_metrics = pd.DataFrame()
    topic_metrics['total_tweets'] = df['topic'].value_counts()
    topic_metrics['positive_ratio'] = topic_sentiment['Positive']
    topic_metrics['negative_ratio'] = topic_sentiment['Negative']
    topic_metrics['sentiment_volatility'] = topic_metrics[['positive_ratio', 'negative_ratio']].std(axis=1)
    
    # Plot top 10 topics by sentiment volatility
    plt.figure(figsize=(12, 6))
    top_volatile = topic_metrics.nlargest(10, 'sentiment_volatility')
    sns.barplot(x=top_volatile.index, y='sentiment_volatility', data=top_volatile)
    plt.title('Top 10 Topics by Sentiment Volatility')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_sentiment_volatility.png'))
    plt.close()
    
    return topic_metrics

def analyze_text_patterns(df: pd.DataFrame, output_dir: str):
    """Analyze detailed text patterns and characteristics."""
    # Analyze word frequency patterns
    stop_words = set(stopwords.words('english'))
    
    # Function to get word frequencies
    def get_word_freq(text_series):
        words = ' '.join(text_series).split()
        return Counter(w for w in words if w not in stop_words)
    
    # Get word frequencies by sentiment
    sentiment_words = {sent: get_word_freq(df[df['sentiment'] == sent]['cleaned_text'])
                      for sent in df['sentiment'].unique()}
    
    # Calculate word importance scores
    word_importance = {}
    all_words = set().union(*[set(words.keys()) for words in sentiment_words.values()])
    
    for word in all_words:
        scores = {sent: words.get(word, 0) for sent, words in sentiment_words.items()}
        word_importance[word] = np.std(list(scores.values()))
    
    # Plot top distinctive words
    top_words = pd.Series(word_importance).nlargest(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_words.values, y=top_words.index)
    plt.title('Top 20 Most Distinctive Words Across Sentiments')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distinctive_words.png'))
    plt.close()
    
    return word_importance

def generate_report(df: pd.DataFrame, metrics: Dict, output_dir: str):
    """Generate a detailed analysis report."""
    report = []
    report.append("# Twitter Sentiment Analysis Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dataset Overview
    report.append("\n## Dataset Overview")
    report.append(f"- Total tweets analyzed: {len(df):,}")
    report.append(f"- Number of unique topics: {df['topic'].nunique():,}")
    report.append(f"- Average tweet length: {df['text_length'].mean():.1f} characters")
    
    # Sentiment Distribution
    report.append("\n## Sentiment Distribution")
    sentiment_dist = df['sentiment'].value_counts(normalize=True) * 100
    for sentiment, percentage in sentiment_dist.items():
        report.append(f"- {sentiment}: {percentage:.1f}%")
    
    # VADER Analysis
    report.append("\n## VADER Sentiment Analysis")
    report.append(f"- Overall agreement with manual labels: {metrics['vader_agreement']:.1f}%")
    
    # Topic Analysis
    report.append("\n## Topic Analysis")
    report.append("\n### Top 5 Topics by Tweet Count")
    top_topics = df['topic'].value_counts().head()
    for topic, count in top_topics.items():
        report.append(f"- {topic}: {count:,} tweets")
    
    # Text Pattern Analysis
    report.append("\n## Text Pattern Analysis")
    report.append("\n### Most Distinctive Words")
    top_distinctive = pd.Series(metrics['word_importance']).nlargest(10)
    for word, score in top_distinctive.items():
        report.append(f"- {word}: {score:.3f}")
    
    # Save report
    with open(os.path.join(output_dir, 'analysis_report.md'), 'w') as f:
        f.write('\n'.join(report))

def main():
    # Create output directory
    output_dir = 'data/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read cleaned data
    print("Reading cleaned data...")
    df = pd.read_csv('data/processed/cleaned_twitter_data.csv')
    
    # Perform detailed analyses
    print("Analyzing sentiment patterns...")
    vader_agreement = analyze_sentiment_patterns(df, output_dir)
    
    print("Analyzing topic-sentiment relationships...")
    topic_metrics = analyze_topic_sentiment_relationships(df, output_dir)
    
    print("Analyzing text patterns...")
    word_importance = analyze_text_patterns(df, output_dir)
    
    # Generate report
    print("Generating analysis report...")
    metrics = {
        'vader_agreement': vader_agreement,
        'topic_metrics': topic_metrics,
        'word_importance': word_importance
    }
    generate_report(df, metrics, output_dir)
    
    print("\nAnalysis complete! Check the following files:")
    print(f"- Detailed report: {os.path.join(output_dir, 'analysis_report.md')}")
    print("- Visualizations in the analysis directory")

if __name__ == "__main__":
    main() 