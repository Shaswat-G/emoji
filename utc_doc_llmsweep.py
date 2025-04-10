import os
import yaml
import json
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import re
import csv
from openai import OpenAI

# General File reads
def load_file(file_path, encoding='utf-8'):
    """Load text from a file with error handling."""
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)

# Configuration and API functions
def load_config(config_path='config.yml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit(1)

def load_api_key(api_key_path):
    """Load API key from file."""
    try:
        with open(api_key_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file '{api_key_path}' not found.")
        exit(1)

def call_llm_api(client, config, prompt):
    """Call LLM API with proper error handling using an existing client."""
    try:
        response_format = {"type": config["response_format"]} if config["response_format"] == "json_object" else None
        messages = [{"role": config["role"], "content": prompt}]
        
        response = client.chat.completions.create(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
            response_format=response_format,
            messages=messages
        )
        
        # Extract content, handling JSON if needed
        if config["response_format"] == "json_object":
            try:
                content = json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                content = response.choices[0].message.content.strip()
        else:
            content = response.choices[0].message.content.strip()
        
        # Calculate token usage and cost
        tokens = dict(response.usage)
        rates = {'gpt-4o-mini-2024-07-18': {"input": 15e-8, "output": 60e-8}}
        model_rates = rates.get(config["model"], {"input": 0, "output": 0})
        cost = tokens['prompt_tokens'] * model_rates["input"] + tokens['completion_tokens'] * model_rates["output"]
        
        return content, tokens, cost, None
    
    except Exception as e:
        return None, None, 0, str(e)

# Document processing functions
def read_file_content(doc_num, folder_path):
    """Read content from a text file with error handling."""
    doc_num = doc_num.replace("/", "_")
    file_path = os.path.join(folder_path, f"{doc_num}.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return ""

def create_extraction_prompt(text, prompt_template):
    """Create a prompt for entity extraction."""
    # Limit text length to avoid token limits
    max_chars = 12000  # Approximate limit to keep within token limits
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    
    return prompt_template.format(input_text=text)

def process_row(row_data, client, config, prompt_template, folder_path):
    """Process a single row with error handling."""
    idx, row = row_data
    result = {
        "idx": idx,
        "emoji_relevant": False,  # New field from prompt.txt
        "people": [],
        "emoji_references": [],
        "entities": [],
        "summary": "",
        "description": "",
        "other_details": "",  # New field from prompt.txt
        "processing_error": None,
        "token_usage": None,
        "api_cost": 0
    }
    
    if pd.isnull(row.get("error_message", "")):
        try:
            # Read text file content but don't store it in the result
            text = read_file_content(row["doc_num"], folder_path)
            
            if text and len(text.strip()) > 0:
                # Create prompt for LLM
                prompt = create_extraction_prompt(text, prompt_template)
                
                # Call LLM API with existing client
                content, tokens, cost, error = call_llm_api(client, config, prompt)
                
                if error:
                    result["processing_error"] = f"API error: {error}"
                elif content:
                    # Extract the required fields from the API response
                    if isinstance(content, dict):
                        result["emoji_relevant"] = content.get("emoji_relevant", False)
                        result["people"] = content.get("people", [])
                        result["emoji_references"] = content.get("emoji_references", [])
                        result["entities"] = content.get("entities", [])
                        result["summary"] = content.get("summary", "")
                        result["description"] = content.get("description", "")
                        result["other_details"] = content.get("other_details", "")
                    else:
                        result["processing_error"] = "Response format error: Expected JSON object"
                    
                    # Store token usage and cost information
                    result["token_usage"] = tokens
                    result["api_cost"] = cost
                else:
                    result["processing_error"] = "No content returned from API"
            else:
                result["processing_error"] = "Empty or invalid text content"
                
        except Exception as e:
            result["processing_error"] = f"Processing error: {str(e)}"
    
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
        text_columns = ["summary", "description", "other_details", "processing_error"]
        for col in text_columns:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(lambda x: remove_illegal_characters(x) if x is not None else None)
        
        # Handle list/array columns separately
        list_columns = ["people", "emoji_references", "entities"]
        for col in list_columns:
            if col in save_df.columns:
                # Convert lists to strings for Excel compatibility
                save_df[col] = save_df[col].apply(
                    lambda x: str(remove_illegal_characters(x)) if x is not None else None
                )
        
        # Convert token usage to string
        if "token_usage" in save_df.columns:
            save_df["token_usage"] = save_df["token_usage"].apply(
                lambda x: str(x) if x is not None else None
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


if __name__ == "__main__":

    multiprocessing.freeze_support()

    config = load_config()
    api_key = load_api_key(config["api_key_path"])
    prompt_template = load_file(config["prompt_path"])
    
    client = OpenAI(api_key=api_key)
    
    base_path = os.getcwd()
    folder_name = "extracted_texts"
    folder_path = os.path.join(base_path, folder_name)
    
    file_name = "utc_register_with_text.xlsx"
    file_path = os.path.join(base_path, file_name)
    
    print("Loading data...")
    df = pd.read_excel(file_path)
    
    df["emoji_relevant"] = False
    df["people"] = None
    df["emoji_references"] = None
    df["entities"] = None
    df["summary"] = None
    df["description"] = None
    df["other_details"] = None
    df["processing_error"] = None
    df["token_usage"] = None
    df["api_cost"] = 0.0
    
    batch_size = 10
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    total_cost = 0.0
    print(f"Processing {total_rows} rows in {num_batches} batches...")
    
    # Use ThreadPoolExecutor for I/O bound operations
    for i in tqdm(range(0, total_rows, batch_size)):
        batch_end = min(i+batch_size, total_rows)
        batch_df = df.iloc[i:batch_end]
        
        # Create list of (idx, row) pairs
        row_data = [(idx, row) for idx, row in batch_df.iterrows()]
        
        # Process rows with ThreadPoolExecutor, passing the client to each worker
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_row, rd, client, config, prompt_template, folder_path) for rd in row_data]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"A worker thread failed: {str(e)}")
        
        # Update the main dataframe with the batch results
        batch_cost = 0.0
        for result in results:
            idx = result["idx"]
            # No longer storing text in the dataframe
            df.at[idx, "emoji_relevant"] = result["emoji_relevant"]
            df.at[idx, "people"] = result["people"]
            df.at[idx, "emoji_references"] = result["emoji_references"]
            df.at[idx, "entities"] = result["entities"]
            df.at[idx, "summary"] = result["summary"]
            df.at[idx, "description"] = result["description"]
            df.at[idx, "other_details"] = result["other_details"]
            df.at[idx, "processing_error"] = result["processing_error"]
            df.at[idx, "token_usage"] = result["token_usage"]
            df.at[idx, "api_cost"] = result["api_cost"]
            batch_cost += result["api_cost"]
        
        total_cost += batch_cost
        print(f"Batch cost: ${batch_cost:.4f} | Running total: ${total_cost:.4f}")
        
        # Save intermediate results every 3 batches
        if (i // batch_size) % 3 == 0 and i > 0:
            interim_file_name = f"utc_register_with_llm_extraction_interim_{i}.xlsx"
            interim_path = os.path.join(base_path, interim_file_name)
            safe_save_to_excel(df, interim_path)
    
    # Save final results
    output_file_name = "utc_register_with_llm_extraction.xlsx"
    output_path = os.path.join(base_path, output_file_name)
    if safe_save_to_excel(df, output_path):
        print(f"Processing complete. Results saved successfully.")
    else:
        print(f"Processing complete but encountered issues saving results.")
    
    print(f"Total API cost: ${total_cost:.4f}")