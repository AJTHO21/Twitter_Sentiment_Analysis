import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import Dict, List, Tuple
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

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

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling various data quality issues."""
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean text data
    df_clean['cleaned_text'] = df_clean['text'].apply(clean_text)
    
    # Remove empty or very short texts
    df_clean = df_clean[df_clean['cleaned_text'].str.len() > 2]
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['cleaned_text'])
    
    # Remove rows with NaN values
    df_clean = df_clean.dropna()
    
    return df_clean

def generate_word_clouds(df: pd.DataFrame, output_dir: str):
    """Generate word clouds for each sentiment category."""
    stop_words = set(stopwords.words('english'))
    
    for sentiment in df['sentiment'].unique():
        # Combine all text for this sentiment
        text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
        
        # Create and generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            stopwords=stop_words,
            max_words=100
        ).generate(text)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {sentiment} Tweets')
        plt.savefig(os.path.join(output_dir, f'wordcloud_{sentiment.lower()}.png'))
        plt.close()

def analyze_text_length_by_category(df: pd.DataFrame, output_dir: str):
    """Analyze and visualize text length distributions by sentiment and topic."""
    # Text length by sentiment
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title('Text Length Distribution by Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_length_by_sentiment.png'))
    plt.close()
    
    # Text length by topic (top 10 topics)
    top_topics = df['topic'].value_counts().head(10).index
    df_top = df[df['topic'].isin(top_topics)]
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='topic', y='text_length', data=df_top)
    plt.title('Text Length Distribution by Topic (Top 10)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_length_by_topic.png'))
    plt.close()

def analyze_common_patterns(df: pd.DataFrame, output_dir: str):
    """Analyze and visualize common patterns in the text data."""
    # Sentiment distribution by topic
    topic_sentiment = pd.crosstab(df['topic'], df['sentiment'], normalize='index') * 100
    
    plt.figure(figsize=(15, 8))
    topic_sentiment.plot(kind='bar', stacked=True)
    plt.title('Sentiment Distribution by Topic (%)')
    plt.xlabel('Topic')
    plt.ylabel('Percentage')
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sentiment_by_topic_percent.png'))
    plt.close()
    
    # Common words by sentiment
    common_words = {}
    for sentiment in df['sentiment'].unique():
        words = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text']).split()
        common_words[sentiment] = Counter(words).most_common(10)
    
    # Plot common words
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for (sentiment, words), ax in zip(common_words.items(), axes.flatten()):
        words, counts = zip(*words)
        sns.barplot(x=list(counts), y=list(words), ax=ax)
        ax.set_title(f'Top 10 Words - {sentiment}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'common_words_by_sentiment.png'))
    plt.close()

def main():
    # Create output directory
    output_dir = 'data/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the data
    print("Reading data...")
    df = pd.read_csv("data/raw/twitter_training.csv", 
                     header=None, 
                     names=['id', 'topic', 'sentiment', 'text'])
    
    # Clean the data
    print("Cleaning data...")
    df_clean = clean_dataset(df)
    
    # Add text length column
    df_clean['text_length'] = df_clean['cleaned_text'].str.len()
    
    # Generate visualizations
    print("Generating word clouds...")
    generate_word_clouds(df_clean, output_dir)
    
    print("Analyzing text length patterns...")
    analyze_text_length_by_category(df_clean, output_dir)
    
    print("Analyzing common patterns...")
    analyze_common_patterns(df_clean, output_dir)
    
    # Save cleaned dataset
    print("Saving cleaned dataset...")
    df_clean.to_csv('data/processed/cleaned_twitter_data.csv', index=False)
    
    # Print summary statistics
    print("\nCleaning Summary:")
    print(f"Original dataset size: {len(df)}")
    print(f"Cleaned dataset size: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} rows")
    print("\nSentiment distribution in cleaned dataset:")
    print(df_clean['sentiment'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    main() 