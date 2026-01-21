import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date, timedelta, datetime
import time
import random
import csv
import os

# --- CONFIGURATION ---
START_DATE = date(2016, 1, 1)
END_DATE = date(2026, 12, 31) 
OUTPUT_FILE = "egx30_full_history_optimized.csv"
BASE_URL = "https://www.dailynewsegypt.com"

# --- NETWORK SETTINGS ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

def get_session():
    """Creates a session with retry logic for stability."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def get_start_date(filename):
    """Reads the CSV to find the last scraped date and resume from the next day."""
    if not os.path.exists(filename):
        return START_DATE
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < 2: return START_DATE
            last_line = lines[-1]
            if ',' not in last_line: return START_DATE
            
            date_str = last_line.split(',')[0].strip().replace('"', '')
            try:
                last_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                return last_date + timedelta(days=1)
            except ValueError:
                return START_DATE
    except Exception:
        return START_DATE

def clean_text(text):
    """Removes newlines and extra spaces."""
    if not text: return ""
    return text.replace('\n', ' ').replace('\r', '').replace('"', "'").strip()

def mine_archives():
    session = get_session()
    current_date = get_start_date(OUTPUT_FILE)
    
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'headline', 'category', 'url'])

    print(f"SYSTEM: Starting high-density scrape from {current_date} to {END_DATE}")

    while current_date <= END_DATE:
        page_num = 1
        day_article_count = 0
        has_next_page = True
        
        while has_next_page:
            if page_num == 1:
                url = f"{BASE_URL}/{current_date.year}/{current_date.month:02d}/{current_date.day:02d}/"
            else:
                url = f"{BASE_URL}/{current_date.year}/{current_date.month:02d}/{current_date.day:02d}/page/{page_num}/"

            try:
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                response = session.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Robust selector for all years
                    articles = soup.find_all('h3', class_='entry-title')

                    current_batch = []
                    for article in articles:
                        link_tag = article.find('a')
                        if not link_tag: continue
                            
                        headline = clean_text(link_tag.get_text())
                        link = link_tag.get('href')
                        
                        # Category extraction
                        category = "General"
                        parent = article.find_parent('div', class_='td_module_wrap') or article.find_parent('div', class_='td-module-thumb')
                        if parent:
                            cat_tag = parent.find('a', class_='td-post-category')
                            if cat_tag: category = clean_text(cat_tag.get_text())
                        
                        current_batch.append([current_date, headline, category, link])

                    if current_batch:
                        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerows(current_batch)
                        day_article_count += len(current_batch)

                    # Pagination Logic
                    if len(current_batch) == 0:
                        has_next_page = False
                    else:
                        page_num += 1
                        # Minimal sleep between pages of the SAME day (speed up)
                        time.sleep(0.2) 

                elif response.status_code == 404:
                    has_next_page = False
                else:
                    has_next_page = False

            except Exception as e:
                print(f"ERROR on {current_date}: {e}")
                has_next_page = False
                time.sleep(2)

        if day_article_count > 0:
            print(f"SUCCESS: {current_date} | {day_article_count} articles")
        else:
            # Minimal logging for empty days
            if current_date.day == 1: print(f"INFO: {current_date} | No data")

        current_date += timedelta(days=1)
        
        # Optimized Speed: 0.5s to 1.2s sleep (Fast but safe)
        time.sleep(random.uniform(0.5, 1.2))

    print("SYSTEM: Complete.")

if __name__ == "__main__":
    mine_archives()
