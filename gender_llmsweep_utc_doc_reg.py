import os
from unittest import result
import yaml
import json
import os
import yaml
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import csv
import logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


import requests
from bs4 import BeautifulSoup
import PyPDF2
import re
from utc_doc_extractor import (
    extract_text_by_extension,
    extract_text_from_html,
    extract_text_from_pdf,
    preprocess_text,
    download_document,
)


from utc_finding_proposals_llm_sweep import (
    load_file,
    load_config,
    load_api_key,
    call_llm_api,
)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
UTC_DOC_REG_COLS = [
    "doc_num",
    "doc_url",
    "subject",
    "date",
    "file_extension",
    "summary",
]


def process_document(row, output_dir):
    """Process a single document: download, extract text, analyze, and save."""
    doc_num = row["doc_num"]
    doc_url = row["doc_url"]
    extension = row["file_extension"]

    error_message = ""

    if not isinstance(doc_url, str) or not extension:
        error_message = "Missing URL or unknown file type"
        print(f"Skipping {doc_num}: {error_message}")
        return {"error_message": error_message}, None

    try:
        # Extract year from doc_num (format: L2/YY-XXX)
        match = re.search(r"L2/(\d{2})-\d+", str(doc_num))
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
            with open(output_file, "w", encoding="utf-8", errors="replace") as f:
                f.write(processed_text)
            return (processed_text, None)
        except Exception as e:
            error_message = f"Error writing text file: {str(e)}"
            print(f"Warning - {error_message} for {doc_num}")
            # Try again with a more aggressive error handling approach
            try:
                cleaned_text = "".join(
                    char if ord(char) < 0x10000 else "?" for char in processed_text
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                error_message += " (Cleaned text was saved)"
                return (cleaned_text, error_message)
            except Exception as e2:
                error_message += f" | Second attempt failed: {str(e2)}"
                print(f"Failed to save text for {doc_num}: {e2}")
                # Create an empty file as a placeholder
                try:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(f"Error processing document {doc_num}: {error_message}")
                except:
                    pass
                return (None, error_message)
    except Exception as e:
        error_message = f"General error processing document {doc_num}: {str(e)}"
        print(error_message)
        return (None, error_message)


def process_row(
    row, processed_text, client, config, system_prompt, user_prompt_template
):

    user_prompt = user_prompt_template.format(
        SUBJECT_LINE=row.get("subject", ""),
        SUMMARY=row.get("summary", ""),
        CONTENTS=processed_text,
    )

    content, tokens, cost, error = call_llm_api(
        client, config, system_prompt, user_prompt
    )

    return content, error


if __name__ == "__main__":

    config_file_name = "config_for_finding_gender_docs.yml"
    config = load_config(os.path.join(BASE_PATH, config_file_name))

    api_key = load_api_key(
        os.path.join(BASE_PATH, config.get("api_key_file_name", "api_key.txt"))
    )
    user_prompt_template = load_file(
        os.path.join(BASE_PATH, config.get("user_prompt_file_name"))
    )
    system_prompt = load_file(
        os.path.join(BASE_PATH, config.get("system_prompt_file_name"))
    )

    client = OpenAI(api_key=api_key)

    file_name = "utc_register_with_llm_extraction.xlsx"
    file_path = os.path.join(BASE_PATH, file_name)
    df = pd.read_excel(file_path, usecols=UTC_DOC_REG_COLS).head(1)

    # Initialize empty columns for API and error fields
    df["gender_expliciteness"] = ""
    df["gender_coverage_pct"] = np.nan
    df["gender_subthemes"] = ""
    df["gender_evidence"] = ""
    df["gender_confidence"] = np.nan
    df["gender_api_error"] = ""
    df["gender_processing_error"] = ""

    output_dir = os.path.join(BASE_PATH, "extracted_texts")
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        # 1) Construct URL, Download File, Process Text
        processed_text, processing_error = process_document(row, output_dir)

        # 2) Call LLM API
        api_response, api_error = None, None
        if processed_text and len(processed_text.strip()) > 0:
            api_response, api_error = process_row(
                row, processed_text, client, config, system_prompt, user_prompt_template
            )

        # Extract fields from API response if available
        if isinstance(api_response, dict):
            df.at[index, "gender_expliciteness"] = api_response.get("expliciteness", "")
            df.at[index, "gender_coverage_pct"] = api_response.get("coverage_pct", np.nan)
            df.at[index, "gender_subthemes"] = str(api_response.get("subthemes", []))
            df.at[index, "gender_evidence"] = str(api_response.get("evidence", []))
            df.at[index, "gender_confidence"] = api_response.get("confidence", np.nan)
        df.at[index, "gender_api_error"] = api_error if api_error else ""
        df.at[index, "gender_processing_error"] = processing_error if processing_error else ""

    # Optionally, save the updated DataFrame
    df.to_excel(os.path.join(BASE_PATH, "gender_llmsweep_results.xlsx"), index=False)
