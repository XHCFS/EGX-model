import pandas as pd
import numpy as np
import os

# ==========================================
# 1. SETUP PATHS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'processed_features.csv')

print(f"Loading raw data from: {data_path}")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"[ERROR] Could not find file at {data_path}")
    exit()

# ==========================================
# 2. REPLICATE PIPELINE LOGIC (CRITICAL)
# ==========================================
print("Reconstructing features from pipeline logic...")

# A. TARGET GENERATION (From Phase 5)
# -----------------------------------
# 1. Calculate Market Return (Mean of all stock log returns)
return_cols = [c for c in df.columns if '_Log_Return' in c]
df['market_return'] = df[return_cols].mean(axis=1)

# 2. Define Target (Top 25% Volatility)
df['volatility_abs'] = df['market_return'].abs()
vol_threshold = df['volatility_abs'].quantile(0.75)
df['target'] = (df['volatility_abs'] > vol_threshold).astype(int)

# B. TECHNICAL FEATURE ENGINEERING
# --------------------------------
# 1. Market Mean RSI
rsi_cols = [c for c in df.columns if '_RSI' in c]
df['Market_Mean_RSI'] = df[rsi_cols].mean(axis=1)

# 2. Market Breadth (% Stocks > SMA50)
sma_cols = [c for c in df.columns if '_SMA_50' in c]
above_sma = pd.DataFrame()
for sma_col in sma_cols:
    stock = sma_col.split('_')[0]
    close_col = f'{stock}_Close'
    if close_col in df.columns:
        above_sma[stock] = (df[close_col] > df[sma_col]).astype(int)
df['Market_Breadth'] = above_sma.mean(axis=1)

# 3. Market Volatility (Dispersion)
df['Market_Volatility'] = df[return_cols].std(axis=1)

# ==========================================
# 3. GENERATE TEXT FOR MANUSCRIPT
# ==========================================
print("\n" + "="*50)
print("   DATASET AUDIT (COPY TO METHODOLOGY PROMPT)")
print("="*50)

print("DATA DESCRIPTION STATS:")
print(f"- Total Samples: {len(df)} trading days")
print(f"- Date Range: {pd.to_datetime(df['date']).min().date()} to {pd.to_datetime(df['date']).max().date()}")
print(f"- Target Definition: High Volatility defined as top 25% of absolute market returns (> {vol_threshold:.4f}).")
print(f"- Class Balance: {df['target'].sum()} High Volatility days ({df['target'].mean():.1%}) vs {len(df)-df['target'].sum()} Stable days.")

# ==========================================
# 4. GENERATE TABLE I (LATEX)
# ==========================================
print("\n" + "="*50)
print("   TABLE I: DESCRIPTIVE STATISTICS (LATEX)")
print("="*50)

# Select the actual features used in the paper
stats_cols = {
    'Market_Mean_RSI': 'Market Mean RSI',
    'Market_Breadth': 'Market Breadth',
    'Market_Volatility': 'Market Volatility (Std)',
    'activity_log': 'News Volume (Log)',
    'mood_financial': 'FinBERT Sentiment',
    'uncertainty': 'Policy Uncertainty',
    'crisis_density': 'Crisis Density'
}

# Filter for existing columns only
valid_cols = [c for c in stats_cols.keys() if c in df.columns]
stats = df[valid_cols].describe().T[['mean', 'std', 'min', 'max']]
stats.index = [stats_cols[c] for c in stats.index]

print(stats.to_latex(float_format="%.2f"))
