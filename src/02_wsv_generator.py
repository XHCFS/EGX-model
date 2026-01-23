"""
World State Vector Generator for Kaggle
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - CORRECT KAGGLE PATH
# ============================================================

INPUT_HEADLINES_PATH = '/kaggle/input/egyptian-news-headlines-2015-2026/egyptian_news_headlines_2015_2026.csv'
OUTPUT_WSV_PATH = '/kaggle/working/world_state_vector.csv'

# ============================================================
# KEYWORD DICTIONARIES
# ============================================================

FINANCIAL_KEYWORDS = {
    'stock', 'share', 'market', 'trading', 'investor', 'investment',
    'egx', 'exchange', 'bourse', 'securities', 'equity', 'portfolio',
    'ipo', 'listing', 'dividend', 'earnings', 'profit', 'loss', 'revenue',
    'bank', 'banking', 'loan', 'credit', 'deposit', 'interest rate',
    'central bank', 'cbe', 'monetary', 'liquidity', 'capital',
    'pound', 'egp', 'dollar', 'currency', 'exchange rate', 'forex',
    'devaluation', 'float', 'remittance', 'gdp', 'inflation', 'economic',
    'economy', 'growth', 'recession', 'unemployment', 'budget', 'imf',
    'company', 'corporate', 'business', 'merger', 'acquisition',
    'oil', 'gas', 'petroleum', 'energy', 'suez canal', 'tourism',
}

POLITICAL_KEYWORDS = {
    'government', 'minister', 'ministry', 'parliament', 'cabinet',
    'president', 'sisi', 'prime minister', 'official', 'law', 'legislation',
    'election', 'vote', 'party', 'opposition', 'political', 'democracy',
    'military', 'army', 'security', 'police', 'terrorism', 'attack',
    'diplomatic', 'embassy', 'ambassador', 'treaty',
    'protest', 'demonstration', 'arrest', 'prison', 'court', 'trial',
}

CRISIS_KEYWORDS = {
    'crisis', 'crash', 'collapse', 'panic', 'emergency', 'disaster',
    'turmoil', 'chaos', 'plunge', 'plummet', 'surge', 'shock', 'default',
    'bankruptcy', 'war', 'conflict', 'attack', 'bomb', 'explosion',
    'death', 'killed', 'violence', 'protest', 'riot', 'revolution',
    'devaluation', 'shortage',
}

URGENCY_KEYWORDS = {
    'breaking', 'urgent', 'alert', 'just in', 'developing',
    'exclusive', 'confirmed', 'immediate', 'emergency', 'critical',
}

# ============================================================
# SENTIMENT MODELS
# ============================================================

print("Loading VADER...")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

print("Loading FinBERT...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
finbert_model = finbert_model.to(device)
finbert_model.eval()
print(f"FinBERT loaded on: {device}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def classify_headline(headline, url=''):
    text = (headline + ' ' + str(url)).lower()
    financial_count = sum(1 for kw in FINANCIAL_KEYWORDS if kw in text)
    political_count = sum(1 for kw in POLITICAL_KEYWORDS if kw in text)
    
    if financial_count >= 2 or (financial_count >= 1 and political_count == 0):
        return 'financial'
    elif political_count >= 2 or (political_count >= 1 and financial_count == 0):
        return 'political'
    elif financial_count > political_count:
        return 'financial'
    elif political_count > financial_count:
        return 'political'
    return 'general'

def get_vader_sentiment(text):
    scores = vader.polarity_scores(str(text))
    return scores['compound']

def get_finbert_batch(texts, batch_size=32):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i:i+batch_size]]
        inputs = finbert_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        for j in range(len(batch)):
            all_scores.append(float(probs[j, 0] - probs[j, 1]))
    return all_scores

def count_keywords(text, keywords):
    text = str(text).lower()
    return sum(1 for kw in keywords if kw in text)

# ============================================================
# MAIN PROCESSING
# ============================================================

print("\nLoading headlines...")
df = pd.read_csv(INPUT_HEADLINES_PATH)
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} headlines")

print("\nClassifying headlines...")
df['category_classified'] = df.apply(lambda r: classify_headline(r['headline'], r.get('url', '')), axis=1)
print(df['category_classified'].value_counts())

print("\nComputing VADER sentiment for all headlines...")
df['vader'] = df['headline'].apply(get_vader_sentiment)

print("\nComputing FinBERT sentiment for financial headlines...")
financial_mask = df['category_classified'] == 'financial'
financial_headlines = df.loc[financial_mask, 'headline'].tolist()
print(f"Processing {len(financial_headlines)} financial headlines...")

if len(financial_headlines) > 0:
    finbert_scores = get_finbert_batch(financial_headlines)
    df.loc[financial_mask, 'finbert'] = finbert_scores
else:
    df['finbert'] = 0.0

df['finbert'] = df['finbert'].fillna(0.0)

print("\nAggregating daily World State Vector...")
daily_groups = df.groupby(df['date'].dt.date)

wsv_records = []
total_days = len(daily_groups)
for idx, (date, group) in enumerate(daily_groups):
    if (idx + 1) % 500 == 0:
        print(f"  Processing day {idx+1}/{total_days}")
    
    headlines = group['headline'].tolist()
    categories = group['category_classified'].tolist()
    vader_scores = group['vader'].tolist()
    finbert_scores = group['finbert'].tolist()
    
    financial_vader = [v for v, c in zip(vader_scores, categories) if c == 'financial']
    political_vader = [v for v, c in zip(vader_scores, categories) if c == 'political']
    financial_finbert = [f for f, c in zip(finbert_scores, categories) if c == 'financial']
    
    wsv = {'date': date}
    
    # Mood
    wsv['mood_general'] = np.mean(vader_scores) if vader_scores else 0.0
    wsv['mood_general_std'] = np.std(vader_scores) if len(vader_scores) > 1 else 0.0
    wsv['mood_financial'] = np.mean(financial_finbert) if financial_finbert else 0.0
    wsv['mood_financial_std'] = np.std(financial_finbert) if len(financial_finbert) > 1 else 0.0
    wsv['mood_political'] = np.mean(political_vader) if political_vader else 0.0
    
    # Uncertainty
    wsv['uncertainty'] = np.std(vader_scores) if len(vader_scores) > 1 else 0.0
    wsv['dispersion'] = (max(vader_scores) - min(vader_scores)) if len(vader_scores) > 1 else 0.0
    
    # Activity
    wsv['activity_total'] = len(headlines)
    wsv['activity_log'] = np.log1p(len(headlines))
    wsv['activity_financial'] = sum(1 for c in categories if c == 'financial')
    wsv['activity_political'] = sum(1 for c in categories if c == 'political')
    
    # Ratios
    total = len(headlines) if len(headlines) > 0 else 1
    wsv['ratio_financial'] = sum(1 for c in categories if c == 'financial') / total
    wsv['ratio_political'] = sum(1 for c in categories if c == 'political') / total
    
    # Crisis & Urgency
    crisis_counts = [count_keywords(h, CRISIS_KEYWORDS) for h in headlines]
    urgency_counts = [count_keywords(h, URGENCY_KEYWORDS) for h in headlines]
    wsv['crisis_total'] = sum(crisis_counts)
    wsv['crisis_density'] = sum(crisis_counts) / total
    wsv['urgency_total'] = sum(urgency_counts)
    
    # Negativity
    wsv['negative_ratio'] = sum(1 for v in vader_scores if v < -0.05) / total
    wsv['positive_ratio'] = sum(1 for v in vader_scores if v > 0.05) / total
    
    wsv_records.append(wsv)

wsv_df = pd.DataFrame(wsv_records)
wsv_df['date'] = pd.to_datetime(wsv_df['date'])
wsv_df = wsv_df.sort_values('date').reset_index(drop=True)

# Add temporal features
print("\nAdding temporal features...")
for col in ['mood_general', 'mood_financial']:
    wsv_df[f'{col}_momentum_1d'] = wsv_df[col] - wsv_df[col].shift(1)
    wsv_df[f'{col}_momentum_3d'] = wsv_df[col] - wsv_df[col].shift(3)
    wsv_df[f'{col}_momentum_7d'] = wsv_df[col] - wsv_df[col].shift(7)

wsv_df['sentiment_shock'] = wsv_df['mood_general'].diff().abs()
wsv_df['activity_surge'] = wsv_df['activity_total'] / wsv_df['activity_total'].rolling(7, min_periods=1).mean()
wsv_df['crisis_rolling_7d'] = wsv_df['crisis_total'].rolling(7, min_periods=1).sum()

# Save
wsv_df.to_csv(OUTPUT_WSV_PATH, index=False)

print("\n" + "="*60)
print("WORLD STATE VECTOR COMPLETE")
print("="*60)
print(f"Output: {OUTPUT_WSV_PATH}")
print(f"Days: {len(wsv_df)}")
print(f"Features: {len(wsv_df.columns) - 1}")
print(f"Date range: {wsv_df['date'].min()} to {wsv_df['date'].max()}")
print("\nSample data:")
print(wsv_df.head())
