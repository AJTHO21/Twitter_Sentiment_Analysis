import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import Dict, List, Tuple

def analyze_text_patterns(text: str) -> Dict[str, int]:
    """Analyze common patterns in text."""
    if pd.isna(text) or not isinstance(text, str):
        return {
            'urls': 0,
            'mentions': 0,
            'hashtags': 0,
            'emojis': 0,
            'numbers': 0,
            'special_chars': 0
        }
    
    patterns = {
        'urls': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(text))),
        'mentions': len(re.findall(r'@\w+', str(text))),
        'hashtags': len(re.findall(r'#\w+', str(text))),
        'emojis': len(re.findall(r'[\U0001F300-\U0001F9FF]', str(text))),
        'numbers': len(re.findall(r'\d+', str(text))),
        'special_chars': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', str(text)))
    }
    return patterns

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Analyze potential data quality issues."""
    issues = {
        'empty_texts': [],
        'very_short_texts': [],
        'duplicate_texts': [],
        'potential_spam': [],
        'non_string_values': []
    }
    
    # Check for non-string values
    non_string_mask = ~df['text'].apply(lambda x: isinstance(x, str))
    issues['non_string_values'] = df[non_string_mask]['text'].tolist()
    
    # Convert non-string values to strings for further analysis
    df['text'] = df['text'].apply(lambda x: str(x) if not pd.isna(x) else '')
    
    # Check for empty or very short texts
    empty_mask = df['text'].str.strip().str.len() == 0
    issues['empty_texts'] = df[empty_mask]['text'].tolist()
    
    # Check for very short texts (less than 3 characters)
    short_mask = df['text'].str.len() < 3
    issues['very_short_texts'] = df[short_mask]['text'].tolist()
    
    # Check for duplicates
    duplicates = df[df.duplicated(subset=['text'], keep=False)]
    issues['duplicate_texts'] = duplicates['text'].tolist()
    
    # Check for potential spam (high frequency of special characters or URLs)
    spam_pattern = df['text'].apply(lambda x: len(re.findall(r'http[s]?://|@\w+|#\w+', str(x))) > 3)
    issues['potential_spam'] = df[spam_pattern]['text'].tolist()
    
    return issues

def plot_distributions(df: pd.DataFrame, save_path: str = None):
    """Create visualizations of the data distributions."""
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Sentiment Distribution
    plt.subplot(2, 2, 1)
    sentiment_counts = df['sentiment'].value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xticks(rotation=45)
    
    # 2. Topic Distribution
    plt.subplot(2, 2, 2)
    topic_counts = df['topic'].value_counts().head(10)
    sns.barplot(x=topic_counts.values, y=topic_counts.index)
    plt.title('Top 10 Topics')
    
    # 3. Text Length Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='text_length', bins=50)
    plt.title('Text Length Distribution')
    
    # 4. Sentiment by Topic
    plt.subplot(2, 2, 4)
    topic_sentiment = pd.crosstab(df['topic'], df['sentiment'])
    sns.heatmap(topic_sentiment, cmap='YlOrRd', annot=True, fmt='d')
    plt.title('Sentiment Distribution by Topic')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def analyze_training_data(file_path: str):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None, 
                     names=['id', 'topic', 'sentiment', 'text'])
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total number of rows: {len(df)}")
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    print("\nTopic Distribution:")
    print(df['topic'].value_counts().head())
    
    # Text length statistics
    df['text_length'] = df['text'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)
    print("\nText Length Statistics:")
    print(df['text_length'].describe())
    
    # Analyze text patterns
    print("\n=== Text Pattern Analysis ===")
    pattern_counts = df['text'].apply(analyze_text_patterns).apply(pd.Series).mean()
    print("\nAverage patterns per tweet:")
    for pattern, count in pattern_counts.items():
        print(f"{pattern}: {count:.2f}")
    
    # Data quality analysis
    print("\n=== Data Quality Issues ===")
    quality_issues = analyze_data_quality(df)
    for issue_type, examples in quality_issues.items():
        print(f"\n{issue_type}:")
        print(f"Count: {len(examples)}")
        if examples:
            print("Examples:")
            for example in examples[:3]:  # Show first 3 examples
                print(f"- {example}")
    
    # Create visualizations
    print("\n=== Generating Visualizations ===")
    plot_distributions(df, 'data/analysis/distributions.png')
    
    # Sample of different sentiments
    print("\n=== Sample Tweets by Sentiment ===")
    for sentiment in df['sentiment'].unique():
        print(f"\n{sentiment} tweets:")
        sample = df[df['sentiment'] == sentiment].sample(min(3, len(df[df['sentiment'] == sentiment])))
        for _, row in sample.iterrows():
            print(f"Topic: {row['topic']}")
            print(f"Text: {row['text']}")
            print("---")

if __name__ == "__main__":
    # Create analysis directory if it doesn't exist
    import os
    os.makedirs('data/analysis', exist_ok=True)
    analyze_training_data("data/raw/twitter_training.csv") 