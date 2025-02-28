import requests
from bs4 import BeautifulSoup
import pandas as pd
import calendar
import re
import os
import logging
import unicodedata
from datetime import datetime
from requests.exceptions import RequestException
from tqdm import tqdm  # For progress bar

# Setup logging
logging.basicConfig(
    filename='unicode_archive_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_datetime_from_id(message_id):
    """Extract timestamp from message ID format like '01401599194.608'"""
    if not message_id:
        return None
    try:
        # Extract the timestamp part (before the dot)
        timestamp_str = message_id.split('.')[0]
        if timestamp_str.startswith('0'):  # Remove leading zero
            timestamp_str = timestamp_str[1:]
        # Convert to datetime
        return datetime.fromtimestamp(int(timestamp_str))
    except (ValueError, IndexError):
        return None

def sanitize_text(text):
    """Clean text to avoid Unicode encoding issues"""
    if not text or not isinstance(text, str):
        return text
        
    try:
        # Normalize Unicode (NFC form tends to work best)
        text = unicodedata.normalize('NFC', text)
        
        # Remove or replace problematic characters
        # Replace surrogate pairs with replacement character
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        return text
    except Exception:
        # If all else fails, return a safe string
        return "[Encoding Error]"

def parse_email_archive(html_content, year, month):
    """Parse the email archive HTML and extract thread data"""
    # Use 'html.parser' instead of default parser for better compatibility
    soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
    emails_data = []
    
    # Look for the message count text to identify the right section
    message_count_tag = soup.find(string=re.compile(r'Messages:\s+\d+'))
    if message_count_tag:
        # Extract the message count
        count_match = re.search(r'Messages:\s+(\d+)', message_count_tag)
        total_messages = int(count_match.group(1)) if count_match else 0
        print(f"Expected message count from HTML: {total_messages}")
    
    # Find ALL li elements that represent emails (direct approach)
    # These li elements have an <a> with href and a following <a> with name attribute
    all_lis = soup.find_all('li')
    
    processed = 0
    for li in all_lis:
        # An email entry must have:
        # 1. An anchor with href (subject link)
        # 2. An anchor with name attribute (message id)
        # 3. An i tag (author)
        href_anchor = li.find('a', href=True)
        if not href_anchor or href_anchor.find_parent('b'):
            continue
        
        # Find the name anchor (for message ID)
        name_anchor = li.find('a', attrs={'name': True})
        if not name_anchor:
            continue
            
        # Find the author tag
        author_tag = li.find('i')
        if not author_tag:
            continue
            
        # This is an email entry
        subject = sanitize_text(href_anchor.get_text(strip=True))
        author = sanitize_text(author_tag.get_text(strip=True))
        msg_id = name_anchor['name']
        
        # Extract timestamp
        time = extract_datetime_from_id(msg_id)
        
        # Extract thread information from HTML comments
        # Comments like <!--0 01401599194.608- --> contain thread info
        thread = subject  # Default to subject
        
        # Try to find the thread from comment
        prev = li.previous
        while prev and not (hasattr(prev, 'string') and prev.string and '<!--' in prev.string):
            prev = prev.previous
            
        if prev and hasattr(prev, 'string'):
            # Comments like <!--0 01401599194.608- -->
            # The number after <!-- indicates thread depth
            comment_match = re.search(r'<!--(\d+)', prev.string)
            if comment_match:
                depth = int(comment_match.group(1))
                # Top-level emails (depth 0) start their own thread
                if depth == 0:
                    thread = subject
        
        # Add to our dataset with sanitized text
        emails_data.append({
            'year': year,
            'month': month,
            'author': author,
            'thread': sanitize_text(thread),
            'subject': subject,
            'time': time,
            'msg_id': msg_id
        })
        processed += 1
    
    print(f"Found {processed} email records")
    
    # Group emails into threads based on subject
    # Emails with the same subject are usually part of the same thread
    subject_to_thread = {}
    for i, email in enumerate(emails_data):
        if email['subject'] not in subject_to_thread:
            subject_to_thread[email['subject']] = email['subject']
        emails_data[i]['thread'] = subject_to_thread[email['subject']]
    
    # Create DataFrame and sort by time
    df = pd.DataFrame(emails_data)
    if not df.empty and 'time' in df.columns:
        df = df.sort_values('time')
    
    return df

def fetch_and_parse_archive(year, month):
    """Fetch and parse a specific archive by year and month"""
    month_str = calendar.month_name[month]
    archive_identifier = f"{year}-{month_str}"
    logging.info(f"Processing archive: {archive_identifier}")
    
    # Build the URL for this archive
    url = f"https://corp.unicode.org/pipermail/unicode/{archive_identifier}/thread.html"
    
    try:
        # Try to fetch the archive
        response = requests.get(url, timeout=45)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Parse the HTML and extract email data
        df = parse_email_archive(response.content, year, month)
        if not df.empty:
            logging.info(f"Successfully extracted {len(df)} emails from {archive_identifier}")
            return df
        else:
            logging.warning(f"No emails found in {archive_identifier}")
            return pd.DataFrame()
            
    except RequestException as e:
        # Handle request errors (404, 500, connection issues, etc.)
        logging.error(f"Failed to fetch archive {archive_identifier}: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        # Handle any other unexpected errors
        logging.error(f"Error processing {archive_identifier}: {str(e)}")
        return pd.DataFrame()

def process_all_archives(start_year=2014, end_year=2021):
    """Process all archives between start_year and end_year"""
    all_emails = pd.DataFrame()
    total_archives = (end_year - start_year + 1) * 12
    
    # Create a progress bar
    with tqdm(total=total_archives, desc="Processing archives") as pbar:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Skip future months if we're in the current year
                if year == datetime.now().year and month > datetime.now().month:
                    pbar.update(1)
                    continue
                
                try:
                    # Process this month's archive
                    df = fetch_and_parse_archive(year, month)
                    
                    # Append to the main dataframe
                    if not df.empty:
                        all_emails = pd.concat([all_emails, df], ignore_index=True)
                        
                        # Save incremental backup every 10 archives - with error handling
                        if (year * 12 + month) % 10 == 0:
                            backup_file = f"unicode_emails_backup_{year}_{month}.csv"
                            try:
                                all_emails.to_csv(backup_file, index=False, encoding='utf-8', errors='replace')
                                logging.info(f"Created backup: {backup_file} with {len(all_emails)} emails")
                            except Exception as e:
                                logging.error(f"Error saving backup file: {str(e)}")
                                # Try with explicit encoding and error handling
                                all_emails.to_csv(backup_file, index=False, encoding='utf-8', 
                                                errors='backslashreplace')
                    
                except Exception as e:
                    logging.error(f"Unexpected error processing {year}-{month}: {str(e)}")
                
                # Update the progress bar
                pbar.update(1)
    
    return all_emails

def main():
    try:
        logging.info("Starting Unicode email archive extraction")
        
        # Directory to store output - fix the path format
        output_dir = "C:/Users/shasw/emoji/archives"  # Remove the leading slash
        # Alternative using raw strings and backslashes
        # output_dir = r"C:\Users\shasw\emoji\archives"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all archives
        all_emails = process_all_archives(2014, 2021)
        
        # Save the final result with explicit encoding and error handling
        output_file = os.path.join(output_dir, "unicode_emails_2014_2021.csv")
        try:
            all_emails.to_csv(output_file, index=False, encoding='utf-8', errors='replace')
        except AttributeError:
            # pandas to_csv doesn't directly support the errors parameter
            # Use alternative approach
            with open(output_file, 'w', encoding='utf-8', errors='backslashreplace') as f:
                all_emails.to_csv(f, index=False)
        
        # Print summary
        print(f"Extraction complete. Total emails: {len(all_emails)}")
        print(f"Results saved to: {output_file}")
        print(f"Check the log file for details: unicode_archive_extraction.log")
        
        # Log summary
        logging.info(f"Extraction complete. Total emails: {len(all_emails)}")
        logging.info(f"Results saved to: {output_file}")
        
        # Display some statistics
        emails_by_year = all_emails.groupby('year').size()
        print("\nEmails by year:")
        print(emails_by_year)
        logging.info(f"Emails by year: {emails_by_year.to_dict()}")
        
    except Exception as e:
        logging.error(f"Fatal error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")
        print("Check the log file for details: unicode_archive_extraction.log")

if __name__ == "__main__":
    main()