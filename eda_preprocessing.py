import pandas as pd
import numpy as np
import warnings
import re
warnings.filterwarnings('ignore')

print("Loading Dataset...")
# Sentiment140 dataset columns
cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)

# target: 0 = negative, 4 = positive
print("Original Dataset Shape:", df.shape)
print(df['target'].value_counts())

# Sampling to 10,000 rows (5,000 pos and 5,000 neg)
print("Sampling 10,000 rows for faster processing...")
df_neg = df[df['target'] == 0].sample(n=5000, random_state=42)
df_pos = df[df['target'] == 4].sample(n=5000, random_state=42)
df_sample = pd.concat([df_neg, df_pos]).reset_index(drop=True)

print("Sampled Dataset Shape:", df_sample.shape)
print(df_sample['target'].value_counts())

print("\nPerforming Basic EDA...")
print("Checking for null values:")
print(df_sample.isnull().sum())

# Changing target values: 0 -> Negative, 4 -> Positive
df_sample['sentiment'] = df_sample['target'].map({0: 0, 4: 1})

# Text cleaning function
def clean_text(text):
    text = text.lower() # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # remove URLs
    text = re.sub(r'\@\w+|\#', '', text) # remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # remove punctuations and numbers
    text = text.strip()
    return text

print("\nStarting Preprocessing Text (Lowercasing, Link Removals, Char cleaning)...")
df_sample['clean_text'] = df_sample['text'].apply(clean_text)

# Drop empty cleaned texts if any
df_sample = df_sample[df_sample['clean_text'] != '']

print("\nFinal shape after cleaning:", df_sample.shape)

# Selecting relevant columns
final_df = df_sample[['clean_text', 'sentiment']]
final_df.to_csv('cleaned_sample.csv', index=False)

print("\nPreprocessing complete. Saved cleaned data to 'cleaned_sample.csv'.")
