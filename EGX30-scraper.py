import yfinance as yf
import pandas as pd
import ta
import os
from datetime import datetime

# Configuration
start_date = "2015-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')
output_folder = "egx_stock_data"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Complete EGX30 Constituents (Yahoo Finance extension: .CA)
tickers = [
    "ABUK.CA",  # Abu Qir Fertilizers
    "ADIB.CA",  # Abu Dhabi Islamic Bank
    "AMOC.CA",  # Alexandria Mineral Oils
    "ARCC.CA",  # Arabian Cement
    "BTFH.CA",  # Beltone Financial
    "CCAP.CA",  # Qalaa Holdings
    "CIEB.CA",  # Credit Agricole Egypt
    "COMI.CA",  # CIB
    "EAST.CA",  # Eastern Company
    "EFIH.CA",  # E-finance
    "EGAL.CA",  # Egypt Aluminum
    "EMFD.CA",  # Emaar Misr
    "ETEL.CA",  # Telecom Egypt
    "FWRY.CA",  # Fawry
    "GBCO.CA",  # GB Corp
    "HRHO.CA",  # EFG Holding
    "ISPH.CA",  # Ibnsina Pharma
    "JUFO.CA",  # Juhayna Food Industries
    "MASR.CA",  # Madinet Masr
    "MCQE.CA",  # Misr Cement
    "MFPC.CA",  # Mopco
    "ORAS.CA",  # Orascom Construction
    "ORHD.CA",  # Orascom Development
    "ORWE.CA",  # Oriental Weavers
    "PHDC.CA",  # Palm Hills
    "RAYA.CA",  # Raya Holding
    "RMDA.CA",  # Rameda
    "SKPC.CA",  # Sidi Kerir Petrochemicals
    "TMGH.CA",  # TMG Holding
    "VLMR.CA"   # Valmore Holding
]

print(f"Downloading data for {len(tickers)} companies...")

for ticker in tickers:
    try:
        # Download historical data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if df.empty:
            print(f"Skipping {ticker}: No data found.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Technical Indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()

        df.dropna(inplace=True)

        # Save
        file_name = f"{ticker.replace('.CA', '')}_price.csv"
        file_path = os.path.join(output_folder, file_name)
        df.to_csv(file_path)
        
        print(f"Processed {ticker}: {len(df)} rows.")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

print(f"Complete. Data saved to {output_folder}/")
