import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TwitterDataset(Dataset):
    """Dataset class for Twitter sentiment analysis."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer: Any, max_length: int):
        """
        Initialize the dataset.
        
        Args:
            texts (List[str]): List of text samples
            labels (List[int]): List of labels
            tokenizer: Tokenizer from transformers
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentClassifier(nn.Module):
    """Transformer-based sentiment classifier."""
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the pre-trained model
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Model output logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[0][:, 0, :]  # Use [CLS] token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class SentimentAnalyzer:
    """Main class for sentiment analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sentiment analyzer.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model = None
        self.tokenizer = None
        self.num_classes = 3  # Positive, Negative, Neutral
        
        # Set random seed
        torch.manual_seed(config['training']['random_seed'])
        np.random.seed(config['training']['random_seed'])

    def setup_model(self):
        """Set up the model and tokenizer."""
        logger.info("Setting up model and tokenizer...")
        
        model_config = self.config['model']['transformer']
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        self.model = SentimentClassifier(
            model_name=model_config['model_name'],
            num_classes=self.num_classes
        ).to(self.device)
        
        logger.info("Model and tokenizer setup completed")

    def create_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation dataloaders.
        
        Args:
            X_train (np.ndarray): Training texts
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation texts
            y_val (np.ndarray): Validation labels
            
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation dataloaders
        """
        logger.info("Creating dataloaders...")
        
        train_dataset = TwitterDataset(
            texts=X_train,
            labels=y_train,
            tokenizer=self.tokenizer,
            max_length=self.config['model']['transformer']['max_length']
        )
        
        val_dataset = TwitterDataset(
            texts=X_val,
            labels=y_val,
            tokenizer=self.tokenizer,
            max_length=self.config['model']['transformer']['max_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['transformer']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['model']['transformer']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )
        
        logger.info("Dataloaders created successfully")
        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training dataloader
            val_loader (DataLoader): Validation dataloader
        """
        logger.info("Starting model training...")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['model']['transformer']['learning_rate'],
            weight_decay=self.config['model']['transformer']['weight_decay']
        )
        
        criterion = nn.CrossEntropyLoss()
        num_epochs = self.config['model']['transformer']['num_epochs']
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clipping']
                )
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            val_loss = self.evaluate(val_loader, criterion)
            
            logger.info(f'Epoch {epoch + 1}:')
            logger.info(f'Average training loss: {avg_train_loss:.4f}')
            logger.info(f'Validation loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                         f"{self.config['output']['model_dir']}/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed")

    def evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader (DataLoader): Validation dataloader
            criterion (nn.Module): Loss function
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_steps += 1
        
        return val_loss / val_steps

    def predict(self, text: str) -> Tuple[int, float]:
        """
        Make a prediction for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            Tuple[int, float]: Predicted class and confidence score
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config['model']['transformer']['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence 