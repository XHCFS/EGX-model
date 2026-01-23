"""
Comprehensive Egyptian News Scraper
=====================================
Collects ALL headlines from multiple sources from 2015 to present.
NO filtering - raw, unbiased collection.

Sources:
1. Daily News Egypt (date-based archives)
2. Egypt Independent (date-based archives)
3. Enterprise (pagination-based)

Output CSV columns:
- date: Publication date (YYYY-MM-DD)
- headline: Full headline text
- source: Source name
- category: Category/section if available
- url: Full article URL

Usage:
    python scrape_all_news.py

Note: This will take several hours to complete due to rate limiting.
      You can interrupt (Ctrl+C) and resume - progress is saved.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import re
import os
import json

# ============================================================
# CONFIGURATION
# ============================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'en-US,en;q=0.9',
}

OUTPUT_FILE = 'all_news_headlines_raw.csv'
CHECKPOINT_FILE = 'scraping_checkpoint.json'
DELAY_BETWEEN_REQUESTS = 0.5  # seconds - balanced speed/safety

# Date range
START_YEAR = 2015
START_MONTH = 1
END_YEAR = 2026
END_MONTH = 1

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_checkpoint(data):
    """Save progress checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_checkpoint():
    """Load progress checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

# Reusable session for connection pooling
_session = None

def get_session():
    """Get session for connection pooling."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
    return _session

def safe_request(url, timeout=20, max_retries=3):
    """Make a request with retries and connection pooling."""
    session = get_session()
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                return None
            else:
                time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"      Timeout, retry {attempt + 1}")
            time.sleep(2)
        except Exception as e:
            print(f"      Error: {e}, retry {attempt + 1}")
            time.sleep(2)
    return None

def extract_date_from_url(url):
    """Extract date from URL."""
    match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
    if match:
        return f'{match.group(1)}-{match.group(2)}-{match.group(3)}'
    return None

def get_months_range():
    """Generate list of (year, month) tuples from start to end."""
    months = []
    year, month = START_YEAR, START_MONTH
    while (year, month) <= (END_YEAR, END_MONTH):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months

def save_intermediate(headlines, filename):
    """Save intermediate results."""
    if headlines:
        df = pd.DataFrame(headlines)
        df.to_csv(filename, index=False)

# ============================================================
# DAILY NEWS EGYPT SCRAPER
# ============================================================

def scrape_dailynewsegypt():
    """
    Scrape Daily News Egypt using date-based archives.
    Archive structure: dailynewsegypt.com/YYYY/MM/page/N/
    """
    headlines = []
    checkpoint = load_checkpoint()
    completed_months = set(checkpoint.get('dne_completed', []))
    
    print("\n" + "="*60)
    print("SCRAPING: DAILY NEWS EGYPT")
    print("="*60)
    
    months = get_months_range()
    
    for year, month in months:
        month_key = f"{year}-{month:02d}"
        
        if month_key in completed_months:
            print(f"  {month_key}: Already scraped, skipping")
            continue
        
        month_headlines = []
        page = 1
        consecutive_empty = 0
        
        while consecutive_empty < 3:
            try:
                if page == 1:
                    url = f"https://www.dailynewsegypt.com/{year}/{month:02d}/"
                else:
                    url = f"https://www.dailynewsegypt.com/{year}/{month:02d}/page/{page}/"
                
                response = safe_request(url)
                if not response:
                    consecutive_empty += 1
                    page += 1
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all article links for this month
                page_count = 0
                all_links = soup.find_all('a', href=True)
                
                seen_headlines = set(h['headline'] for h in month_headlines)
                
                for link in all_links:
                    href = link.get('href', '')
                    headline = link.get_text(strip=True)
                    
                    # Filter to articles from this month
                    if f'dailynewsegypt.com/{year}/{month:02d}/' not in href:
                        continue
                    if not headline or len(headline) < 20:
                        continue
                    if headline.lower().startswith('daily news'):
                        continue
                    if headline in seen_headlines:
                        continue
                    
                    date = extract_date_from_url(href)
                    if not date:
                        continue
                    
                    # Detect category from URL
                    category = 'General'
                    if '/business/' in href.lower():
                        category = 'Business'
                    elif '/economy/' in href.lower():
                        category = 'Economy'
                    elif '/politics/' in href.lower():
                        category = 'Politics'
                    elif '/sports/' in href.lower():
                        category = 'Sports'
                    elif '/opinion/' in href.lower():
                        category = 'Opinion'
                    
                    month_headlines.append({
                        'date': date,
                        'headline': headline,
                        'source': 'DailyNewsEgypt',
                        'category': category,
                        'url': href
                    })
                    seen_headlines.add(headline)
                    page_count += 1
                
                if page_count == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                
                page += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
            except KeyboardInterrupt:
                print("\n  Interrupted! Saving progress...")
                headlines.extend(month_headlines)
                checkpoint['dne_completed'] = list(completed_months)
                save_checkpoint(checkpoint)
                save_intermediate(headlines, 'dne_headlines_partial.csv')
                return headlines
            except Exception as e:
                print(f"      Error on page {page}: {e}")
                page += 1
        
        headlines.extend(month_headlines)
        completed_months.add(month_key)
        print(f"  {month_key}: {len(month_headlines)} headlines (total: {len(headlines)})")
        
        # Save checkpoint every month
        checkpoint['dne_completed'] = list(completed_months)
        save_checkpoint(checkpoint)
        
        # Save CSV every 10 months for safety
        if len(completed_months) % 10 == 0:
            save_intermediate(headlines, 'dne_headlines_partial.csv')
            print(f"    [Checkpoint saved: {len(headlines)} headlines to dne_headlines_partial.csv]")
    
    print(f"\nDaily News Egypt total: {len(headlines)} headlines")
    save_intermediate(headlines, 'dne_headlines.csv')
    return headlines

# ============================================================
# EGYPT INDEPENDENT SCRAPER
# ============================================================

def scrape_egypt_independent():
    """
    Scrape Egypt Independent using date-based archives (sequential, reliable).
    """
    headlines = []
    checkpoint = load_checkpoint()
    completed_months = set(checkpoint.get('ei_completed', []))
    
    print("\n" + "="*60)
    print("SCRAPING: EGYPT INDEPENDENT")
    print("="*60)
    
    months = get_months_range()
    
    for year, month in months:
        month_key = f"{year}-{month:02d}"
        
        if month_key in completed_months:
            print(f"  {month_key}: Already scraped, skipping")
            continue
        
        month_headlines = []
        page = 1
        consecutive_empty = 0
        
        while consecutive_empty < 3:
            try:
                if page == 1:
                    url = f"https://egyptindependent.com/{year}/{month:02d}/"
                else:
                    url = f"https://egyptindependent.com/{year}/{month:02d}/page/{page}/"
                
                response = safe_request(url)
                if not response:
                    consecutive_empty += 1
                    page += 1
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find h2 tags with article links
                page_count = 0
                seen_headlines = set(h['headline'] for h in month_headlines)
                
                h2_tags = soup.find_all('h2')
                for h2 in h2_tags:
                    link = h2.find('a')
                    if not link:
                        continue
                    
                    headline = link.get_text(strip=True)
                    href = link.get('href', '')
                    
                    if not headline or len(headline) < 15:
                        continue
                    if headline in seen_headlines:
                        continue
                    
                    # Get date - EI uses text dates in sibling div.thumb-meta or .date class
                    date = None
                    
                    # Method 1: Look for date in parent container
                    parent = h2.parent
                    if parent:
                        date_elem = parent.find(class_='date')
                        if not date_elem:
                            date_elem = parent.find(class_='thumb-meta')
                        if date_elem:
                            date_text = date_elem.get_text(strip=True)
                            # Parse "July 31, 2015" format
                            date_match = re.search(
                                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                                date_text, re.I
                            )
                            if date_match:
                                months_map = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
                                         'july':7,'august':8,'september':9,'october':10,'november':11,'december':12}
                                mon = months_map.get(date_match.group(1).lower(), 1)
                                day = int(date_match.group(2))
                                yr = int(date_match.group(3))
                                date = f"{yr:04d}-{mon:02d}-{day:02d}"
                    
                    # Method 2: Try URL (some older articles might have dates)
                    if not date:
                        date = extract_date_from_url(href)
                    
                    # Skip if no date found
                    if not date:
                        continue
                    
                    # Check date is in target month (but be lenient - archive pages may have mixed dates)
                    if not date.startswith(f"{year}-{month:02d}"):
                        continue
                    
                    # Get category
                    category = 'General'
                    cat_elem = h2.find_previous(class_=re.compile(r'cat|category'))
                    if cat_elem:
                        cat_link = cat_elem.find('a')
                        if cat_link:
                            category = cat_link.get_text(strip=True)
                    
                    month_headlines.append({
                        'date': date,
                        'headline': headline,
                        'source': 'EgyptIndependent',
                        'category': category,
                        'url': href
                    })
                    seen_headlines.add(headline)
                    page_count += 1
                
                if page_count == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                
                page += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)
                
            except KeyboardInterrupt:
                print("\n  Interrupted! Saving progress...")
                headlines.extend(month_headlines)
                checkpoint['ei_completed'] = list(completed_months)
                save_checkpoint(checkpoint)
                save_intermediate(headlines, 'ei_headlines_partial.csv')
                return headlines
            except Exception as e:
                print(f"      Error on page {page}: {e}")
                page += 1
        
        headlines.extend(month_headlines)
        completed_months.add(month_key)
        print(f"  {month_key}: {len(month_headlines)} headlines (total: {len(headlines)})")
        
        checkpoint['ei_completed'] = list(completed_months)
        save_checkpoint(checkpoint)
        
        # Save CSV every 10 months for safety
        if len(completed_months) % 10 == 0:
            save_intermediate(headlines, 'ei_headlines_partial.csv')
            print(f"    [Checkpoint saved: {len(headlines)} headlines]")
    
    print(f"\nEgypt Independent total: {len(headlines)} headlines")
    save_intermediate(headlines, 'ei_headlines.csv')
    return headlines

# ============================================================
# ENTERPRISE SCRAPER
# ============================================================

def scrape_enterprise():
    """
    Scrape Enterprise using pagination (sequential, reliable).
    """
    headlines = []
    checkpoint = load_checkpoint()
    start_page = checkpoint.get('enterprise_page', 1)
    
    print("\n" + "="*60)
    print("SCRAPING: ENTERPRISE")
    print("="*60)
    
    page = start_page
    consecutive_empty = 0
    oldest_date_seen = None
    
    while consecutive_empty < 5:
        try:
            if page == 1:
                url = "https://enterpriseam.com/egypt/"
            else:
                url = f"https://enterpriseam.com/egypt/page/{page}/"
            
            response = safe_request(url, timeout=30)
            if not response:
                consecutive_empty += 1
                page += 1
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = soup.find_all('article')
            page_count = 0
            
            for article in articles:
                title_elem = article.find(['h2', 'h3', 'h4'])
                if not title_elem:
                    continue
                
                link = title_elem.find('a')
                if not link:
                    continue
                
                headline = link.get_text(strip=True)
                href = link.get('href', '')
                
                if not headline or len(headline) < 10:
                    continue
                
                # Get date
                date = None
                time_elem = article.find('time')
                if time_elem:
                    datetime_attr = time_elem.get('datetime')
                    if datetime_attr:
                        date = datetime_attr[:10]
                
                if not date:
                    date = extract_date_from_url(href)
                
                if not date:
                    # Try text-based date
                    date_text = article.find(class_=re.compile(r'date|time|posted|meta'))
                    if date_text:
                        text = date_text.get_text()
                        date_match = re.search(
                            r'(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(\d{4})',
                            text, re.I
                        )
                        if date_match:
                            months_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                                     'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
                            day = int(date_match.group(1))
                            mon = months_map.get(date_match.group(2).lower()[:3], 1)
                            yr = int(date_match.group(3))
                            date = f"{yr:04d}-{mon:02d}-{day:02d}"
                
                if headline and date:
                    headlines.append({
                        'date': date,
                        'headline': headline,
                        'source': 'Enterprise',
                        'category': 'Business',
                        'url': href
                    })
                    page_count += 1
                    
                    if oldest_date_seen is None or date < oldest_date_seen:
                        oldest_date_seen = date
            
            if page_count == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
            
            if page % 10 == 0:
                print(f"  Page {page}: {page_count} headlines (oldest: {oldest_date_seen})")
            
            # Check if we've gone far enough back
            if oldest_date_seen and oldest_date_seen < '2015-01-01':
                print(f"  Reached 2015, stopping at page {page}")
                break
            
            # Save checkpoint
            if page % 50 == 0:
                checkpoint['enterprise_page'] = page
                save_checkpoint(checkpoint)
                save_intermediate(headlines, 'enterprise_headlines_partial.csv')
            
            page += 1
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
        except KeyboardInterrupt:
            print("\n  Interrupted! Saving progress...")
            checkpoint['enterprise_page'] = page
            save_checkpoint(checkpoint)
            save_intermediate(headlines, 'enterprise_headlines_partial.csv')
            return headlines
        except Exception as e:
            print(f"  Page {page}: Error - {e}")
            page += 1
    
    print(f"\nEnterprise total: {len(headlines)} headlines")
    save_intermediate(headlines, 'enterprise_headlines.csv')
    return headlines

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main function - scrape all sources and combine."""
    
    print("="*60)
    print("COMPREHENSIVE EGYPTIAN NEWS SCRAPER")
    print("="*60)
    print(f"Target date range: {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Press Ctrl+C to pause - progress will be saved!")
    print("="*60)
    
    all_headlines = []
    
    # Check if we already have DNE data
    DNE_EXISTING_FILE = 'EGX30_News_Headlines.csv'
    if os.path.exists(DNE_EXISTING_FILE):
        print("\n" + "="*60)
        print("SKIPPING: DAILY NEWS EGYPT")
        print(f"  (Already have data in {DNE_EXISTING_FILE})")
        print("="*60)
        try:
            dne_df = pd.read_csv(DNE_EXISTING_FILE)
            dne_headlines = []
            for _, row in dne_df.iterrows():
                dne_headlines.append({
                    'date': row['date'],
                    'headline': row['headline'],
                    'source': 'DailyNewsEgypt',
                    'category': row.get('category', 'General'),
                    'url': row.get('url', '')
                })
            all_headlines.extend(dne_headlines)
            print(f"  Loaded {len(dne_headlines)} existing DNE headlines")
        except Exception as e:
            print(f"  Error loading DNE: {e}, will scrape instead")
            dne_headlines = scrape_dailynewsegypt()
            all_headlines.extend(dne_headlines)
    else:
        try:
            dne_headlines = scrape_dailynewsegypt()
            all_headlines.extend(dne_headlines)
        except Exception as e:
            print(f"Daily News Egypt error: {e}")
    
    try:
        ei_headlines = scrape_egypt_independent()
        all_headlines.extend(ei_headlines)
    except Exception as e:
        print(f"Egypt Independent error: {e}")
    
    try:
        enterprise_headlines = scrape_enterprise()
        all_headlines.extend(enterprise_headlines)
    except Exception as e:
        print(f"Enterprise error: {e}")
    
    # Combine and save
    if not all_headlines:
        print("\nNo headlines collected!")
        return
    
    df = pd.DataFrame(all_headlines)
    
    # Clean
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.drop_duplicates(subset=['headline', 'date', 'source'])
    df = df.sort_values('date')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETE")
    print("="*60)
    print(f"Total headlines: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nBy source:")
    print(df['source'].value_counts().to_string())
    print(f"\nBy category:")
    print(df['category'].value_counts().to_string())
    print(f"\nBy year:")
    df['year'] = pd.to_datetime(df['date']).dt.year
    print(df['year'].value_counts().sort_index().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
