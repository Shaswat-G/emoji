# -----------------------------------------------------------------------------
# Script: utc_doc_summarizer.py
# Summary: Extracts keywords and summaries from UTC document texts using RAKE,
#          YAKE, and LSA, saving results to an enriched Excel file for
#          downstream analysis.
# Inputs:  utc_register_with_text.xlsx (document metadata and text),
#          extracted_texts/ (directory of document text files)
# Outputs: utc_register_with_text_and_summary.xlsx (enriched with keywords/summaries)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------

import os
import yake
import pandas as pd
from rake_nltk import Rake
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
import re  # Add this import for regex
import csv  # Add this import for CSV handling
import numpy as np  # Add this import

# Move function definitions here (outside the main block)
def read_file_content(doc_num, folder_path):
    doc_num = doc_num.replace("/", "_")
    file_path = os.path.join(folder_path, f"{doc_num}.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return ""

def get_rake_keywords(text, proportion=0.01):
    if not text or len(text.strip()) == 0:
        return []
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords[:int(max(1, len(keywords) * proportion))]

def get_yake_keywords(text, proportion=0.01):
    if not text or len(text.strip()) == 0:
        return []
    yake_ext = yake.KeywordExtractor(lan="en", n=3)
    keywords = yake_ext.extract_keywords(text)
    return keywords[:int(max(1, len(keywords) * proportion))]

def get_lsa_summary(text, sentences_count=5):
    if not text or len(text.strip()) == 0:
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=min(sentences_count, len(parser.document.sentences)))
    return " ".join(str(sentence) for sentence in summary)

def process_row(row_data):
    """Process a single row with error handling"""
    idx, row = row_data
    result = {
        "idx": idx,
        "text": "",
        "rake_keywords": None,
        "yake_keywords": None, 
        "lsa_summary": None,
        "processing_error": None
    }
    
    if pd.isnull(row["error_message"]):
        try:
            text = read_file_content(row["doc_num"], folder_path)
            result["text"] = text
            
            try:
                result["rake_keywords"] = get_rake_keywords(text, 0.01)
            except Exception as e:
                result["processing_error"] = f"Rake error: {str(e)}"
                
            try:
                result["yake_keywords"] = get_yake_keywords(text, 0.01)
            except Exception as e:
                if result["processing_error"]:
                    result["processing_error"] += f"; Yake error: {str(e)}"
                else:
                    result["processing_error"] = f"Yake error: {str(e)}"
                
            try:
                result["lsa_summary"] = get_lsa_summary(text, 5)
            except Exception as e:
                if result["processing_error"]:
                    result["processing_error"] += f"; LSA error: {str(e)}"
                else:
                    result["processing_error"] = f"LSA error: {str(e)}"
                
        except Exception as e:
            result["processing_error"] = f"File reading error: {str(e)}"
    
    return result

def remove_illegal_characters(text):
    """Remove characters that are not allowed in Excel worksheets with encoding checks."""
    if not isinstance(text, str):
        # If it's a list or array, convert each element
        if isinstance(text, (list, np.ndarray, pd.Series)):
            return [remove_illegal_characters(str(item)) for item in text]
        return text
        
    # Try to re-encode as UTF-8 to catch any encoding issues
    try:
        text = text.encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        text = str(text.encode('ascii', errors='replace').decode('ascii'))
    
    # Remove control characters and other problematic characters for Excel
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)

def save_to_csv(df, file_path):
    """Save DataFrame to CSV with UTF-8 encoding."""
    try:
        # Create a copy to avoid modifying the original dataframe
        save_df = df.copy()
        
        # Convert all complex objects to strings for CSV
        for col in save_df.columns:
            save_df[col] = save_df[col].apply(
                lambda x: str(x) if x is not None else None
            )
            
        save_df.to_csv(file_path, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        return True
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")
        return False

def safe_save_to_excel(df, file_path, csv_backup=True):
    """Try to save to Excel, fall back to CSV if it fails."""
    try:
        # Create a copy to avoid modifying the original dataframe
        save_df = df.copy()
        
        # Sanitize all text columns
        print("Sanitizing data before saving...")
        text_columns = ["text", "lsa_summary", "processing_error"]
        for col in text_columns:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(lambda x: remove_illegal_characters(x) if x is not None else None)
        
        # Handle list/array columns separately
        list_columns = ["rake_keywords", "yake_keywords"]
        for col in list_columns:
            if col in save_df.columns:
                # Convert lists to strings for Excel compatibility
                save_df[col] = save_df[col].apply(
                    lambda x: str(remove_illegal_characters(x)) if x is not None else None
                )
        
        # Try to save as Excel
        save_df.to_excel(file_path, index=False, engine='openpyxl')
        print(f"Successfully saved to Excel: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving to Excel: {str(e)}")
        if csv_backup:
            csv_path = file_path.replace('.xlsx', '.csv')
            if save_to_csv(df, csv_path):
                print(f"Saved backup to CSV: {csv_path}")
                return True
        return False

# Main execution block
if __name__ == "__main__":
    # Add freeze_support for Windows
    multiprocessing.freeze_support()
    
    base_path = os.getcwd()
    folder_name = "extracted_texts"
    folder_path = os.path.join(base_path, folder_name)
    
    file_name = "utc_register_with_text.xlsx"
    file_path = os.path.join(base_path, file_name)
    
    print("Loading data...")
    df = pd.read_excel(file_path)  # Process all rows
    
    df["rake_keywords"] = None
    df["yake_keywords"] = None
    df["lsa_summary"] = None
    df["processing_error"] = None
    
    batch_size = 50  # Smaller batch size
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    print(f"Processing {total_rows} rows in {num_batches} batches...")
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    for i in tqdm(range(0, total_rows, batch_size)):
        batch_end = min(i+batch_size, total_rows)
        batch_df = df.iloc[i:batch_end]
        
        # Create list of (idx, row) pairs
        row_data = [(idx, row) for idx, row in batch_df.iterrows()]
        
        # Process rows with ThreadPoolExecutor
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_row, rd) for rd in row_data]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"A worker thread failed: {str(e)}")
        
        # Update the main dataframe with the batch results
        for result in results:
            idx = result["idx"]
            df.at[idx, "text"] = result["text"]
            df.at[idx, "rake_keywords"] = result["rake_keywords"]
            df.at[idx, "yake_keywords"] = result["yake_keywords"]
            df.at[idx, "lsa_summary"] = result["lsa_summary"]
            df.at[idx, "processing_error"] = result["processing_error"]
        
        # Save intermediate results every 5 batches
        if (i // batch_size) % 5 == 0 and i > 0:
            interim_file_name = f"utc_register_with_text_and_summary_interim_{i}.xlsx"
            interim_path = os.path.join(base_path, interim_file_name)
            safe_save_to_excel(df, interim_path)
    
    # Save final results
    output_file_name = "utc_register_with_text_and_summary.xlsx"
    output_path = os.path.join(base_path, output_file_name)
    if safe_save_to_excel(df, output_path):
        print(f"Processing complete. Results saved successfully.")
    else:
        print(f"Processing complete but encountered issues saving results.")
