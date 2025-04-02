import pandas as pd
import numpy as np
import os
import re
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji  # Add emoji library for detecting and handling emojis
from tqdm import tqdm  # Import tqdm for progress bar

# Compile regex patterns once for better performance
UNICODE_PATTERNS = [
    re.compile(r'[Uu]\+([0-9A-F]{4,6})'),             # U+1F600 format
    re.compile(r'\\u([0-9A-F]{4,6})'),                # \u1F600 format
    re.compile(r'&#x([0-9A-F]{4,6});'),               # &#x1F600; HTML hex entity
    re.compile(r'\\x{([0-9A-F]{4,6})}'),              # \x{1F600} format
    re.compile(r'code[\s-]?point[\s:]+([0-9A-F]{4,6})'),  # "code point 1F600"
]
SHORTCODE_PATTERN = re.compile(r':([a-z0-9_\-+]+):')
DOC_REF_PATTERN = re.compile(r'(L2/\d{2}[-‐–—−]\d{3})')

stop_words = set(stopwords.words('english'))

# Define comprehensive emoji-related keywords for relevance detection
EMOJI_KEYWORDS = [
    "emoji", "emojis", "zwj", "emoticon", "emoticons", "kaomoji", "afroji", "animoji",
    "emojipedia", "emojify", "emojification", "pictograph", "pictographic", "smiley",
    "smileys", "smilies", "face with", "faces with", "emoji sequence", "emoji character",
    "emoji proposal", "emoji encoding", "unicode emoji", "skin tone", "face-smiling",
    "face-with", "hand-gesture", "flag emoji", "keycap", "presentation selector",
    "emoji modifier", "regional indicator", "emoji tag sequence", "emoji variation sequence"
]

def extract_text_from_pdf(content):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_html(content):
    try:
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        # Get text and clean whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        print(f"Error extracting HTML text: {e}")
        return ""

def preprocess_text(text):
    # Standardize different types of dashes and remove extra whitespace
    return re.sub(r'\s+', ' ', re.sub(r'[‐–—−]', '-', text))

def extract_emojis(text):
    if not text:
        return {'emoji_chars': [], 'unicode_points': [], 'emoji_shortcodes': []}
    
    # Extract actual emoji characters
    emoji_list = [emoji_dict['emoji'] for emoji_dict in emoji.emoji_list(text) 
                 if isinstance(emoji_dict, dict) and 'emoji' in emoji_dict]
    
    # Demojize to capture shortcodes
    demojized = emoji.demojize(text)
    
    # Extract Unicode codepoint references
    all_unicode_matches = []
    for pattern in UNICODE_PATTERNS:
        all_unicode_matches.extend(pattern.findall(text))
    
    # Extract emoji shortcodes
    shortcodes = SHORTCODE_PATTERN.findall(demojized)
    
    return {
        'emoji_chars': list(set(emoji_list)),
        'unicode_points': list(set(all_unicode_matches)),
        'emoji_shortcodes': list(set(shortcodes))
    }

def is_emoji_relevant(text):
    """Check if the document is emoji-relevant based on keyword presence."""
    if not text:
        return False, []
    
    text_lower = text.lower()
    found_keywords = []
    
    # Check for each keyword with word boundary detection
    for keyword in EMOJI_KEYWORDS:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_keywords.append(keyword)
    
    return bool(found_keywords), list(set(found_keywords))

def extract_doc_refs(text):
    """Extract document references from the text."""
    if not text:
        return []
    
    # Find document references and standardize dash types
    matches = DOC_REF_PATTERN.findall(text)
    standardized_matches = [re.sub(r'[‐–—−]', '-', match) for match in matches]
    
    return list(sorted(set(standardized_matches)))

def download_document(url, timeout=30):
    """Download document from URL and return response content."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            print(f"Failed to download {url}: Status code {response.status_code}")
            return None
        return response.content
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_text_by_extension(content, extension):
    """Extract text based on file extension."""
    if extension.lower() == 'pdf':
        return extract_text_from_pdf(content)
    elif extension.lower() in ['html', 'htm']:
        return extract_text_from_html(content)
    elif extension.lower() == 'txt':
        return content.decode('utf-8', errors='replace')
    else:
        print(f"Unsupported file extension: {extension}")
        return ""

def process_document(row, output_dir):
    """Process a single document: download, extract text, analyze, and save."""
    doc_num = row['doc_num']
    doc_url = row['doc_url']
    extension = row['file_extension']
    
    if not isinstance(doc_url, str) or not extension:
        print(f"Skipping {doc_num}: Missing URL or unknown file type")
        return {}, None
    
    # Extract year from doc_num (format: L2/YY-XXX)
    match = re.search(r'L2/(\d{2})-\d+', str(doc_num))
    if not match:
        print(f"Could not extract year from {doc_num}")
        return {}, None
    
    year = match.group(1)
    full_url = f"https://www.unicode.org/L2/L20{year}/{doc_url}"
    
    print(f"Downloading: {full_url}")
    content = download_document(full_url)
    if content is None:
        return {}, None
    
    # Extract and process text
    text = extract_text_by_extension(content, extension)
    if not text:
        print(f"No text extracted from {doc_num}")
        return {}, None
    
    processed_text = preprocess_text(text)
    
    # Save processed text to file
    output_file = os.path.join(output_dir, f"{doc_num.replace('/', '_')}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    # Get file size in KB
    file_size_kb = os.path.getsize(output_file) / 1024
    
    # Analyze text for emoji relevance and references
    is_relevant, found_keywords = is_emoji_relevant(processed_text)
    extracted_emojis = extract_emojis(processed_text)
    doc_refs = extract_doc_refs(processed_text)
    
    print(f"Successfully processed {doc_num} (Emoji Relevant: {is_relevant})")
    
    return {
        'extracted_doc_refs': doc_refs,
        'is_emoji_relevant': is_relevant,
        'emoji_keywords_found': found_keywords,
        'emoji_chars': extracted_emojis['emoji_chars'],
        'unicode_points': extracted_emojis['unicode_points'],
        'emoji_shortcodes': extracted_emojis['emoji_shortcodes'],
        'file_size_kb': round(file_size_kb, 2)
    }, processed_text

def main():
    working_dir = os.getcwd()
    file_name = 'utc_register_all_classified.xlsx'
    file_path = os.path.join(working_dir, file_name)
    output_dir = os.path.join(working_dir, "extracted_texts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataframe (remove sample() in production)
    df = pd.read_excel(file_path).sample(10)
    
    # Extract file extensions more robustly
    df["file_extension"] = df["doc_url"].apply(
        lambda x: os.path.splitext(str(x))[1].lstrip('.') if isinstance(x, str) else None
    )
    
    # Initialize columns
    df['extracted_doc_refs'] = "[]"
    df['emoji_chars'] = "[]"
    df['unicode_points'] = "[]"
    df['is_emoji_relevant'] = False
    df['emoji_keywords_found'] = "[]"
    df['emoji_shortcodes'] = "[]"
    df['file_size_kb'] = 0.0
    
    print("Starting text extraction...")
    
    # Calculate documents to process
    docs_to_process = df[pd.notna(df['file_extension'])].shape[0]
    total_time = 0
    processed_count = 0
    
    # Use tqdm for progress tracking
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing documents"):
        if pd.notna(row['file_extension']):
            start_time = time.time()
            
            results, _ = process_document(row, output_dir)
            
            end_time = time.time()
            doc_time = end_time - start_time
            total_time += doc_time
            processed_count += 1
            avg_time = total_time / processed_count
            
            if results:
                for key, value in results.items():
                    df.at[i, key] = value
            
            tqdm.write(f"Document {row['doc_num']} processed in {doc_time:.2f}s (Avg: {avg_time:.2f}s)")
            time.sleep(1)  # Be polite to the server
    
    print(f"Processing completed. Average time per document: {total_time/processed_count:.2f}s")
    df.to_excel(os.path.join(working_dir, "utc_register_with_text.xlsx"), index=False)
    print("Text extraction completed.")

if __name__ == "__main__":
    main()