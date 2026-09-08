import pandas as pd
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def clean_text(text):
    text = text.lower()
    return text

train_df['text'] = train_df['text'].apply(clean_text)
test_df['text'] = test_df['text'].apply(clean_text)

