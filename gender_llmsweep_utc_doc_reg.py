import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from utc_finding_proposals_llm_sweep import (
    load_file,
    load_config,
    load_api_key,
    call_llm_api,
)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ------------------ Configurable Constants ------------------
BATCH_SIZE = 4  # Number of rows per batch
MAX_CHARS = 12000  # Max chars for LLM prompt input
INTERIM_SAVE_INTERVAL = 4  # Save every batch for testing
# Detect number of CPU cores and set number of workers
NUM_WORKERS = 2 * multiprocessing.cpu_count()
# -----------------------------------------------------------

UTC_DOC_REG_COLS = [
    "doc_num",
    "doc_url",
    "subject",
    "date",
    "file_extension",
    "summary",
]


def process_document(row, output_dir):
    """Read already processed document text from file."""
    doc_num = row["doc_num"]

    # Construct the filename for the already processed text file
    text_file = os.path.join(output_dir, f"{doc_num.replace('/', '_')}.txt")

    try:
        # Check if the processed text file exists
        if not os.path.exists(text_file):
            error_message = f"Processed text file not found: {text_file}"
            print(f"Skipping {doc_num}: {error_message}")
            return None, error_message

        # Read the processed text from file
        with open(text_file, "r", encoding="utf-8", errors="replace") as f:
            processed_text = f.read()

        if not processed_text or not processed_text.strip():
            error_message = "Empty text file"
            print(f"Warning - {error_message} for {doc_num}")
            return None, error_message

        print(f"Loaded processed text for: {doc_num}")
        return processed_text, None

    except Exception as e:
        error_message = f"Error reading processed text file for {doc_num}: {str(e)}"
        print(error_message)
        return None, error_message


def process_row(
    row, processed_text, client, config, system_prompt, user_prompt_template
):
    """Process a single row with LLM API call and JSON parsing."""
    try:
        user_prompt = user_prompt_template.format(
            SUBJECT_LINE=row.get("subject", ""),
            SUMMARY=row.get("summary", ""),
            CONTENTS=processed_text,
        )

        content, tokens, cost, error = call_llm_api(
            client, config, system_prompt, user_prompt
        )

        # Try to parse JSON if content is a string
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logging.warning(
                    f"Failed to parse JSON response for doc {row.get('doc_num', 'unknown')}"
                )

        return content, error
    except Exception as e:
        logging.error(
            f"Error processing row for doc {row.get('doc_num', 'unknown')}: {str(e)}"
        )
        return None, str(e)


def process_batch(
    batch_df,
    batch_indices,
    client,
    config,
    system_prompt,
    user_prompt_template,
    output_dir,
):
    """Process a batch of documents in parallel."""
    results = []

    def process_single_document(row_data):
        idx, row = row_data
        doc_num = row.get("doc_num", f"doc_{idx}")

        try:
            # Process document
            processed_text, processing_error = process_document(row, output_dir)

            # Call LLM API if text extraction succeeded
            api_response, api_error = None, None
            if (
                isinstance(processed_text, str)
                and processed_text
                and len(processed_text.strip()) > 0
            ):
                # Truncate text if too long
                if len(processed_text) > MAX_CHARS:
                    processed_text = processed_text[:MAX_CHARS] + "...[truncated]"

                api_response, api_error = process_row(
                    row,
                    processed_text,
                    client,
                    config,
                    system_prompt,
                    user_prompt_template,
                )

            return {
                "idx": idx,
                "doc_num": doc_num,
                "processed_text": processed_text,
                "processing_error": processing_error,
                "api_response": api_response,
                "api_error": api_error,
            }
        except Exception as e:
            logging.error(f"Error processing document {doc_num}: {str(e)}")
            return {
                "idx": idx,
                "doc_num": doc_num,
                "processed_text": None,
                "processing_error": str(e),
                "api_response": None,
                "api_error": None,
            }

    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, len(batch_df))) as executor:
        row_data = [
            (idx, row) for idx, (_, row) in zip(batch_indices, batch_df.iterrows())
        ]
        future_to_row = {
            executor.submit(process_single_document, row_data): row_data
            for row_data in row_data
        }

        for future in as_completed(future_to_row):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                row_data = future_to_row[future]
                logging.error(
                    f"Batch processing failed for document {row_data[1].get('doc_num', 'unknown')}: {str(e)}"
                )
                results.append(
                    {
                        "idx": row_data[0],
                        "doc_num": row_data[1].get("doc_num", "unknown"),
                        "processed_text": None,
                        "processing_error": str(e),
                        "api_response": None,
                        "api_error": None,
                    }
                )

    return results


def save_batch_results(df, batch_results, batch_num, base_path):
    """Save batch results to the main DataFrame and create interim save."""
    for result in batch_results:
        idx = result["idx"]
        api_response = result["api_response"]

        # Extract fields from API response if available
        if isinstance(api_response, dict):
            df.at[idx, "gender_expliciteness"] = api_response.get("expliciteness", "")
            df.at[idx, "gender_coverage_pct"] = api_response.get("coverage_pct", np.nan)
            df.at[idx, "gender_subthemes"] = str(api_response.get("subthemes", []))
            df.at[idx, "gender_evidence"] = str(api_response.get("evidence", []))
            df.at[idx, "gender_confidence"] = api_response.get("confidence", np.nan)

        df.at[idx, "gender_api_error"] = (
            result["api_error"] if result["api_error"] else ""
        )
        df.at[idx, "gender_processing_error"] = (
            result["processing_error"] if result["processing_error"] else ""
        )

    # Save interim results every INTERIM_SAVE_INTERVAL batches
    if batch_num % INTERIM_SAVE_INTERVAL == 0:
        interim_file = os.path.join(
            base_path, f"gender_llmsweep_interim_batch_{batch_num}.xlsx"
        )
        try:
            df.to_excel(interim_file, index=False)
            logging.info(f"Saved interim results to {interim_file}")
        except Exception as e:
            logging.warning(f"Failed to save interim results: {str(e)}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
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
        df = pd.read_excel(file_path, usecols=UTC_DOC_REG_COLS).head(
            20
        )  # Process 20 for testing

        logging.info(f"Loaded {len(df)} documents for processing")
        logging.info(f"Using {NUM_WORKERS} workers for parallel processing")

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

        # Process in batches
        total_rows = len(df)
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

        logging.info(
            f"Processing {total_rows} documents in {num_batches} batches of {BATCH_SIZE}"
        )

        for batch_num in range(num_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_rows)
            batch_indices = list(range(start_idx, end_idx))
            batch_df = df.iloc[start_idx:end_idx]

            logging.info(
                f"Processing batch {batch_num + 1}/{num_batches} (rows {start_idx}-{end_idx-1})"
            )

            # Process batch
            batch_results = process_batch(
                batch_df,
                batch_indices,
                client,
                config,
                system_prompt,
                user_prompt_template,
                output_dir,
            )

            # Save results
            save_batch_results(df, batch_results, batch_num + 1, BASE_PATH)

            logging.info(f"Completed batch {batch_num + 1}/{num_batches}")

        # Save final results
        output_file = os.path.join(BASE_PATH, "gender_llmsweep_results.xlsx")
        df.to_excel(output_file, index=False)
        logging.info(f"Final results saved to: {output_file}")

    except Exception as e:
        logging.error(f"Script failed with error: {str(e)}")
        raise
