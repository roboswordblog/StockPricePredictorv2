import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def clean_text(text):
    text = text.lower()
    return text

train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_text(texts, tokenizer, max_length=128):
    encodings = tokenizer(
        list(texts),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

train_texts, val_texts, train_targets, val_targets = train_test_split(
    train_df['text'].values,
    train_df['target'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df['target'])