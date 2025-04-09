import os
import yake
import pandas as pd
from rake_nltk import Rake
import concurrent.futures
from tqdm import tqdm
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser

base_path = os.getcwd()
folder_name = "extracted_texts"
folder_path = os.path.join(base_path, folder_name)

file_name = "utc_register_with_text.xlsx"
file_path = os.path.join(base_path, file_name)

df = pd.read_excel(file_path)  # Process all rows, not just sample
# df = df.sample(5)  # Remove sampling for production

df["rake_keywords"] = None
df["yake_keywords"] = None
df["lsa_summary"] = None
df["processing_error"] = None  # New column for processing errors

def read_file_content(doc_num, folder_path):
    doc_num = doc_num.replace("/", "_")
    file_path = os.path.join(folder_path, f"{doc_num}.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except:
        return ""

def get_rake_keywords(text, proportion=0.01):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords[:int(len(keywords) * proportion)]

def get_yake_keywords(text, proportion=0.01):
    yake_ext = yake.KeywordExtractor(lan="en", n=3)
    keywords = yake_ext.extract_keywords(text)
    return keywords[:int(len(keywords) * proportion)]

def get_lsa_summary(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=sentences_count)
    return " ".join(str(sentence) for sentence in summary)

def process_row(row):
    """Process a single row with error handling"""
    result = {
        "idx": row.name,
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

def process_batch(batch_df):
    """Process a batch of rows in parallel"""
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_row, row) for _, row in batch_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"A worker process failed: {str(e)}")
    return results

# Process in batches
batch_size = 100
total_rows = len(df)
num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division

print(f"Processing {total_rows} rows in {num_batches} batches...")

for i in tqdm(range(0, total_rows, batch_size)):
    batch_df = df.iloc[i:min(i+batch_size, total_rows)]
    batch_results = process_batch(batch_df)
    
    # Update the main dataframe with the batch results
    for result in batch_results:
        idx = result["idx"]
        df.at[idx, "text"] = result["text"]
        df.at[idx, "rake_keywords"] = result["rake_keywords"]
        df.at[idx, "yake_keywords"] = result["yake_keywords"]
        df.at[idx, "lsa_summary"] = result["lsa_summary"]
        df.at[idx, "processing_error"] = result["processing_error"]
    
    # Save intermediate results every 5 batches
    if (i // batch_size) % 5 == 0 and i > 0:
        interim_file_name = f"utc_register_with_text_and_summary_interim_{i}.xlsx"
        df.to_excel(os.path.join(base_path, interim_file_name), index=False)
        print(f"Saved interim results to {interim_file_name}")

output_file_name = "utc_register_with_text_and_summary.xlsx"
df.to_excel(os.path.join(base_path, output_file_name), index=False)
print(f"Processing complete. Results saved to {output_file_name}")