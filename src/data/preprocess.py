import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with necessary NLTK resources."""
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {e}")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()

    def clean_text(self, text):
        """
        Clean the text by removing special characters, URLs, and converting to lowercase.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def tokenize_and_lemmatize(self, text):
        """
        Tokenize the text and apply lemmatization.
        
        Args:
            text (str): Input text to process
            
        Returns:
            list: List of lemmatized tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return tokens

    def prepare_data(self, df):
        """
        Prepare the dataset by cleaning text and encoding labels.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'text' and 'sentiment' columns
            
        Returns:
            tuple: (processed_texts, encoded_labels)
        """
        logger.info("Starting data preparation...")
        
        # Clean texts
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Tokenize and lemmatize
        df['processed_text'] = df['cleaned_text'].apply(self.tokenize_and_lemmatize)
        
        # Join tokens back into text
        df['processed_text'] = df['processed_text'].apply(lambda x: ' '.join(x))
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(df['sentiment'])
        
        logger.info("Data preparation completed successfully")
        
        return df['processed_text'].values, encoded_labels

def load_data(train_path, val_path):
    """
    Load and prepare the training and validation datasets.
    
    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    logger.info("Loading datasets...")
    
    try:
        # Load datasets
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Prepare training data
        X_train, y_train = preprocessor.prepare_data(train_df)
        
        # Prepare validation data
        X_val, y_val = preprocessor.prepare_data(val_df)
        
        logger.info("Datasets loaded and prepared successfully")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        logger.error(f"Error loading or preparing data: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    train_path = "data/raw/twitter_training.csv"
    val_path = "data/raw/twitter_validation.csv"
    
    try:
        X_train, y_train, X_val, y_val = load_data(train_path, val_path)
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}") 