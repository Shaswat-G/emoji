# -----------------------------------------------------------------------------
# Script: utc_doc_extractor.py
# Summary: Downloads, extracts, and analyzes text from UTC register documents,
#          saving results and metadata for downstream analysis of emoji
#          relevance, references, and document citations.
# Inputs:  utc_register_all_classified.xlsx (document register with URLs)
# Outputs: utc_register_with_text.xlsx (enriched metadata), extracted_texts/
#          (directory of document text files)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import re
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import time
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
        try:
            return content.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error decoding text content: {e}")
            return ""
    else:
        print(f"Unsupported file extension: {extension}")
        return ""

def process_document(row, output_dir):
    """Process a single document: download, extract text, analyze, and save."""
    doc_num = row['doc_num']
    doc_url = row['doc_url']
    extension = row['file_extension']
    
    error_message = ""
    
    if not isinstance(doc_url, str) or not extension:
        error_message = "Missing URL or unknown file type"
        print(f"Skipping {doc_num}: {error_message}")
        return {"error_message": error_message}, None
    
    try:
        # Extract year from doc_num (format: L2/YY-XXX)
        match = re.search(r'L2/(\d{2})-\d+', str(doc_num))
        if not match:
            error_message = f"Could not extract year from document number"
            print(f"{error_message}: {doc_num}")
            return {"error_message": error_message}, None
        
        year = match.group(1)
        full_url = f"https://www.unicode.org/L2/L20{year}/{doc_url}"
        
        print(f"Downloading: {full_url}")
        content = download_document(full_url)
        if content is None:
            error_message = "Failed to download document"
            return {"error_message": error_message}, None
        
        # Extract and process text
        text = extract_text_by_extension(content, extension)
        if not text:
            error_message = "No text extracted from document"
            print(f"{error_message}: {doc_num}")
            return {"error_message": error_message}, None
        
        processed_text = preprocess_text(text)
        
        # Save processed text to file with error handling
        output_file = os.path.join(output_dir, f"{doc_num.replace('/', '_')}.txt")
        try:
            with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                f.write(processed_text)
        except Exception as e:
            error_message = f"Error writing text file: {str(e)}"
            print(f"Warning - {error_message} for {doc_num}")
            # Try again with a more aggressive error handling approach
            try:
                # Replace problematic characters
                cleaned_text = ''.join(char if ord(char) < 0x10000 else '?' for char in processed_text)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                error_message += " (Cleaned text was saved)"
            except Exception as e2:
                error_message += f" | Second attempt failed: {str(e2)}"
                print(f"Failed to save text for {doc_num}: {e2}")
                # Create an empty file as a placeholder
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"Error processing document {doc_num}: {error_message}")
                except:
                    pass
        
        # Get file size in KB
        try:
            file_size_kb = os.path.getsize(output_file) / 1024
        except:
            file_size_kb = 0
            error_message += " | Could not determine file size"
        
        # Analyze text for emoji relevance and references
        try:
            is_relevant, found_keywords = is_emoji_relevant(processed_text)
            extracted_emojis = extract_emojis(processed_text)
            doc_refs = extract_doc_refs(processed_text)
        except Exception as e:
            is_relevant, found_keywords = False, []
            extracted_emojis = {'emoji_chars': [], 'unicode_points': [], 'emoji_shortcodes': []}
            doc_refs = []
            error_message += f" | Error analyzing text: {str(e)}"
        
        print(f"Successfully processed {doc_num} (Emoji Relevant: {is_relevant})")
        
        return {
            'extracted_doc_refs': doc_refs,
            'is_emoji_relevant': is_relevant,
            'emoji_keywords_found': found_keywords,
            'emoji_chars': extracted_emojis['emoji_chars'],
            'unicode_points': extracted_emojis['unicode_points'],
            'emoji_shortcodes': extracted_emojis['emoji_shortcodes'],
            'file_size_kb': round(file_size_kb, 2),
            'error_message': error_message
        }, processed_text
    
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        print(f"Error processing {doc_num}: {error_message}")
        return {"error_message": error_message}, None

def main():
    working_dir = os.getcwd()
    file_name = 'utc_register_all_classified.xlsx'
    file_path = os.path.join(working_dir, file_name)
    output_dir = os.path.join(working_dir, "extracted_texts")
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    results_file = os.path.join(working_dir, "utc_register_with_text.xlsx")
    checkpoint_file = os.path.join(working_dir, "utc_register_with_text_checkpoint.xlsx")
    
    # Check for existing results file to resume processing
    if os.path.exists(results_file):
        print(f"Found existing results file. Loading to continue processing...")
        df = pd.read_excel(results_file)
        # Find which documents have been processed
        processed_mask = df['error_message'].notna() | (df['file_size_kb'] > 0)
        processed_count = processed_mask.sum()
        print(f"Already processed {processed_count} documents.")
    else:
        df = pd.read_excel(file_path)
        processed_count = 0
    
    # Extract file extensions more robustly
    if 'file_extension' not in df.columns:
        # Use np.nan for invalid URLs instead of None to work properly with notna()
        df["file_extension"] = df["doc_url"].apply(
            lambda x: os.path.splitext(str(x))[1].lstrip('.') if isinstance(x, str) and x else np.nan
        )
        # Only allow valid extensions (convert empty strings and unsupported extensions to NaN)
        valid_extensions = ['pdf', 'html', 'htm', 'txt']
        df['file_extension'] = df['file_extension'].apply(
            lambda ext: ext.lower() if isinstance(ext, str) and ext.lower() in valid_extensions else np.nan
        )
    
    # Initialize columns if they don't exist
    for col in ['extracted_doc_refs', 'emoji_chars', 'unicode_points', 'is_emoji_relevant', 
                'emoji_keywords_found', 'emoji_shortcodes', 'file_size_kb', 'error_message']:
        if col not in df.columns:
            if col in ['is_emoji_relevant', 'file_size_kb']:
                df[col] = 0
            elif col == 'error_message':
                df[col] = np.nan  # Use NaN instead of empty string for unprocessed documents
            else:
                df[col] = "[]"
    
    print("Starting text extraction...")
    
    # Calculate documents to process
    docs_to_process = df[pd.notna(df['file_extension'])].shape[0]
    total_time = 0
    batch_size = 100
    
    # Create mask for documents that need processing - fix by checking for empty strings too
    to_process_mask = pd.notna(df['file_extension']) & (
        pd.isna(df['error_message']) | 
        (df['error_message'] == "") & (df['file_size_kb'] == 0)
    )
    to_process_indices = df[to_process_mask].index
    remaining_docs = len(to_process_indices)
    
    print(f"Documents to process: {remaining_docs} out of {docs_to_process} total")
    
    # Process in batches
    batch_num = 0
    for batch_start in range(0, remaining_docs, batch_size):
        batch_num += 1
        batch_end = min(batch_start + batch_size, remaining_docs)
        current_batch_indices = to_process_indices[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_num} ({batch_start+1}-{batch_end} of {remaining_docs})...")
        
        # Process the current batch
        batch_processed = 0
        
        # Use tqdm for progress tracking within the batch
        for i in tqdm(current_batch_indices, desc=f"Batch {batch_num}"):
            row = df.loc[i]
            start_time = time.time()
            
            try:
                results, _ = process_document(row, output_dir)
                
                end_time = time.time()
                doc_time = end_time - start_time
                total_time += doc_time
                batch_processed += 1
                
                if results:
                    for key, value in results.items():
                        df.at[i, key] = value
                
                tqdm.write(f"Document {row['doc_num']} processed in {doc_time:.2f}s")
            except Exception as e:
                df.at[i, 'error_message'] = f"Unhandled exception: {str(e)}"
                tqdm.write(f"Error with document {row['doc_num']}: {e}")
        
        processed_count += batch_processed
        
        # Save checkpoint after each batch
        try:
            # First save to temporary file then rename to avoid corruption
            df.to_excel(checkpoint_file, index=False)
            if os.path.exists(results_file):
                os.replace(checkpoint_file, results_file)  # Atomic replacement
            else:
                os.rename(checkpoint_file, results_file)
            print(f"Checkpoint saved after batch {batch_num}. Total processed: {processed_count}/{docs_to_process}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            # Try a backup save method
            try:
                backup_file = os.path.join(working_dir, f"utc_register_with_text_backup_batch{batch_num}.csv")
                df.to_csv(backup_file, index=False, encoding='utf-8', errors='replace')
                print(f"Backup CSV saved instead: {backup_file}")
            except Exception as e2:
                print(f"Could not save checkpoint: {e2}")
    
    if processed_count > 0:
        print(f"Processing completed. Average time per document: {total_time/processed_count:.2f}s")
    else:
        print("No documents were processed. All may have been processed already.")
    
    # Save final results
    try:
        df.to_excel(results_file, index=False)
        print("Text extraction completed and results saved.")
    except Exception as e:
        print(f"Error saving final results: {e}")
        # Try a backup save method
        try:
            df.to_csv(os.path.join(working_dir, "utc_register_with_text_backup_final.csv"), 
                     index=False, encoding='utf-8', errors='replace')
            print("Backup CSV saved instead.")
        except:
            print("Could not save results. Please check data for invalid characters.")

if __name__ == "__main__":
    main()
