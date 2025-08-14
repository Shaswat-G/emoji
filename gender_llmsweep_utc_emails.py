# -----------------------------------------------------------------------------
# Script: gender_llmsweep_utc_emails.py
# Summary: Analyzes gender-related content in UTC email communications using LLM processing.
#          Processes email subject lines and body content to extract gender discourse
#          patterns, explicitness levels, coverage metrics, and thematic analysis.
# Inputs:  Email workbook with 'subject' and 'body' fields (Excel format),
#          config_for_finding_gender_emails.yml (LLM configuration),
#          system/user prompt files for email gender analysis
# Outputs: gender_llmsweep_email_results.xlsx (final results with gender analysis),
#          gender_llmsweep_email_interim_batch_*.xlsx (interim batch saves)
# Features: Parallel processing with ThreadPoolExecutor, batch processing,
#          interim saves, comprehensive error handling, progress tracking,
#          email-specific text preprocessing and analysis
# Context: Part of emoji proposal research pipeline analyzing gender representation
#          and discourse patterns in Unicode Technical Committee email communications.
# -----------------------------------------------------------------------------

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
BATCH_SIZE = 4  # Number of emails per batch
MAX_CHARS = 12000  # Max chars for LLM prompt input
INTERIM_SAVE_INTERVAL = 50  # Save every batch for testing
# Detect number of CPU cores and set number of workers
NUM_WORKERS = 2 * multiprocessing.cpu_count()
# -----------------------------------------------------------

EMAIL_COLS = [
    "subject",
    "body",
]


def process_email(row):
    """Process a single email: basic validation and prepare for analysis."""
    try:
        subject = row.get("subject", "")
        body = row.get("body", "")

        # Convert None to empty string
        subject = str(subject) if subject is not None else ""
        body = str(body) if body is not None else ""

        if not subject and not body:
            error_message = "Both subject and body are empty"
            print(f"Skipping email: {error_message}")
            return None, error_message

        print(f"Processed email with subject: {subject[:50]}...")
        return True, None  # Just return success flag

    except Exception as e:
        error_message = f"Error processing email: {str(e)}"
        print(error_message)
        return None, error_message


def process_row(row, client, config, system_prompt, user_prompt_template):
    """Process a single row with LLM API call and JSON parsing."""
    try:
        user_prompt = user_prompt_template.format(
            SUBJECT_LINE=row.get("subject", ""),
            BODY=row.get("body", ""),
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
                    f"Failed to parse JSON response for email with subject: {row.get('subject', 'unknown')[:50]}"
                )

        return content, error
    except Exception as e:
        logging.error(
            f"Error processing row for email with subject: {row.get('subject', 'unknown')[:50]}: {str(e)}"
        )
        return None, str(e)


def process_batch(
    batch_df,
    batch_indices,
    client,
    config,
    system_prompt,
    user_prompt_template,
):
    """Process a batch of emails in parallel."""
    results = []

    def process_single_email(row_data):
        idx, row = row_data
        email_identifier = f"email_{idx}_{str(row.get('subject', 'no_subject'))[:30]}"

        try:
            # Process email
            email_valid, processing_error = process_email(row)

            # Call LLM API if email is valid
            api_response, api_error = None, None
            if email_valid:
                api_response, api_error = process_row(
                    row,
                    client,
                    config,
                    system_prompt,
                    user_prompt_template,
                )

            return {
                "idx": idx,
                "email_identifier": email_identifier,
                "processing_error": processing_error,
                "api_response": api_response,
                "api_error": api_error,
            }
        except Exception as e:
            logging.error(f"Error processing email {email_identifier}: {str(e)}")
            return {
                "idx": idx,
                "email_identifier": email_identifier,
                "processing_error": str(e),
                "api_response": None,
                "api_error": None,
            }

    # Process emails in parallel
    with ThreadPoolExecutor(max_workers=min(NUM_WORKERS, len(batch_df))) as executor:
        row_data = [
            (idx, row) for idx, (_, row) in zip(batch_indices, batch_df.iterrows())
        ]
        future_to_row = {
            executor.submit(process_single_email, row_data): row_data
            for row_data in row_data
        }

        for future in as_completed(future_to_row):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                row_data = future_to_row[future]
                logging.error(
                    f"Batch processing failed for email {row_data[1].get('subject', 'unknown')[:30]}: {str(e)}"
                )
                results.append(
                    {
                        "idx": row_data[0],
                        "email_identifier": f"email_{row_data[0]}_error",
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
            base_path, f"gender_llmsweep_email_interim_batch_{batch_num}.xlsx"
        )
        try:
            df.to_excel(interim_file, index=False)
            logging.info(f"Saved interim email results to {interim_file}")
        except Exception as e:
            logging.warning(f"Failed to save interim email results: {str(e)}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        config_file_name = "config_for_finding_gender_emails.yml"
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
        file_name = "utc_email_combined_with_llm_extraction_doc_ref.xlsx"
        file_path = os.path.join(BASE_PATH, file_name)

        df = (pd.read_excel(file_path, usecols=EMAIL_COLS))

        logging.info(f"Loaded {df.shape[0]} emails for processing")
        logging.info(f"Using {NUM_WORKERS} workers for parallel processing")

        # Initialize empty columns for API and error fields
        df["gender_expliciteness"] = ""
        df["gender_coverage_pct"] = np.nan
        df["gender_subthemes"] = ""
        df["gender_evidence"] = ""
        df["gender_confidence"] = np.nan
        df["gender_api_error"] = ""
        df["gender_processing_error"] = ""

        # Process in batches
        total_rows = len(df)
        num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

        logging.info(
            f"Processing {total_rows} emails in {num_batches} batches of {BATCH_SIZE}"
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
            )

            # Save results
            save_batch_results(df, batch_results, batch_num + 1, BASE_PATH)

            logging.info(f"Completed batch {batch_num + 1}/{num_batches}")

        # Save final results
        output_file = os.path.join(BASE_PATH, "gender_llmsweep_email_results.xlsx")
        df.to_excel(output_file, index=False)
        logging.info(f"Final email results saved to: {output_file}")

    except Exception as e:
        logging.error(f"Email processing script failed with error: {str(e)}")
        raise
