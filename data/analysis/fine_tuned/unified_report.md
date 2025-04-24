# Unified Sentiment Analysis Results

## Model: Fine-tuned RoBERTa

## Overall Accuracy: 0.811

## POS_NEG Classifier
Accuracy: 0.833

Classification Report
```
              precision    recall  f1-score   support

    Negative       0.83      0.83      0.83        30
    Positive       0.83      0.83      0.83        30

    accuracy                           0.83        60
   macro avg       0.83      0.83      0.83        60
weighted avg       0.83      0.83      0.83        60

```

## POS_NEU Classifier
Accuracy: 0.850

Classification Report
```
              precision    recall  f1-score   support

     Neutral       0.89      0.80      0.84        30
    Positive       0.82      0.90      0.86        30

    accuracy                           0.85        60
   macro avg       0.85      0.85      0.85        60
weighted avg       0.85      0.85      0.85        60

```

## NEG_NEU Classifier
Accuracy: 0.750

Classification Report
```
              precision    recall  f1-score   support

    Negative       0.78      0.70      0.74        30
     Neutral       0.73      0.80      0.76        30

    accuracy                           0.75        60
   macro avg       0.75      0.75      0.75        60
weighted avg       0.75      0.75      0.75        60

```