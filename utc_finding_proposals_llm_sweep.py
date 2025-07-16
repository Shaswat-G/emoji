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

# ------------------ Configurable Constants ------------------
BATCH_SIZE = 1  # For future batch processing, currently sequential
MAX_CHARS = 12000  # Max chars for LLM prompt input
INTERIM_SAVE_INTERVAL = 100  # Save every N rows

# -----------------------------------------------------------


# General File reads
def load_file(file_path, encoding="utf-8"):
    """Load text from a file with error handling."""
    try:
        with open(file_path, "r", encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        return None


# Configuration and API functions
def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Error: Configuration file '{config_path}' not found.")
        return None


def load_api_key(api_key_path):
    """Load API key from file."""
    try:
        with open(api_key_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"Error: API key file '{api_key_path}' not found.")
        return None


def call_llm_api(client, config, system_prompt, user_prompt):
    """Call LLM API with proper error handling using an existing client."""
    try:
        response_format = (
            {"type": config["response_format"]}
            if config["response_format"] == "json_object"
            else None
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
            response_format=response_format,
            messages=messages,
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
        rates = {"gpt-4o-mini-2024-07-18": {"input": 15e-8, "output": 60e-8}}
        model_rates = rates.get(config["model"], {"input": 0, "output": 0})
        cost = (
            tokens["prompt_tokens"] * model_rates["input"]
            + tokens["completion_tokens"] * model_rates["output"]
        )

        return content, tokens, cost, None

    except Exception as e:
        return None, None, 0, str(e)


def create_extraction_prompt(text: dict, prompt_template):
    """Create a prompt for entity extraction."""
    subject = text.get("subject", "")
    summary = text.get("summary", "")
    description = text.get("description", "")
    old_classifier_answer = text.get("old_classifier_answer", "")

    return prompt_template.format(
        SUBJECT_LINE=subject, SUMMARY=summary, DESCRIPTION=description,
        OLD_CLASSIFIER_ANSWER=old_classifier_answer
    )

    return prompt_template.format(input_text=text)


def process_row(row_data, client, config, system_prompt, user_prompt_template):
    """Process a single row with error handling."""
    idx, row = row_data
    result = {
        "idx": idx,
        "is_emoji_proposal": False,
        "document_classification": None,
        "processing_error": None,
    }

    if pd.isnull(row.get("error_message", "")):
        try:
            text = {
                "subject": row.get("subject", ""),
                "summary": row.get("summary", ""),
                "description": row.get("description", ""),
                "old_classifier_answer": row.get("doc_type", ""),
            }
            user_prompt = create_extraction_prompt(text, user_prompt_template)
            content, tokens, cost, error = call_llm_api(client, config, system_prompt, user_prompt)

            if error:
                result["processing_error"] = f"API error: {error}"
            elif content:
                result["is_emoji_proposal"] = content.get("emoji_proposal", False)
                result["document_classification"] = content.get("labels", None)
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
        text = text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        text = str(text.encode("ascii", errors="replace").decode("ascii"))

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

        save_df.to_csv(file_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
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
        text_columns = ["processing_error"]
        for col in text_columns:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(
                    lambda x: remove_illegal_characters(x) if x is not None else None
                )

        # Convert token usage to string
        if "token_usage" in save_df.columns:
            save_df["token_usage"] = save_df["token_usage"].apply(
                lambda x: str(x) if x is not None else None
            )

        # Try to save as Excel
        save_df.to_excel(file_path, index=False, engine="openpyxl")
        print(f"Successfully saved to Excel: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving to Excel: {str(e)}")
        if csv_backup:
            csv_path = file_path.replace(".xlsx", ".csv")
            if save_to_csv(df, csv_path):
                print(f"Saved backup to CSV: {csv_path}")
                return True
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    config = load_config("config_for_finding_proposals.yml")
    if config is None:
        logging.error("Exiting due to missing config.")
    else:
        api_key = load_api_key(config["api_key_path"])
        if api_key is None:
            logging.error("Exiting due to missing API key.")
        else:
            user_prompt_template = load_file(config["user_prompt_path"])
            system_prompt = load_file(config["system_prompt_path"])
            if user_prompt_template is None:
                logging.error("Exiting due to missing prompt template.")
            else:
                client = OpenAI(api_key=api_key)

                base_path = os.getcwd()
                file_name = "utc_register_with_llm_extraction.xlsx"
                file_path = os.path.join(base_path, file_name)

                logging.info("Loading data...")
                df = pd.read_excel(file_path).sample(20).reset_index(drop=True)

                df["document_classification"] = None
                df["is_emoji_proposal"] = False
                df["processing_error"] = None

                total_rows = df.shape[0]
                logging.info(f"Processing {total_rows} rows")

                # Process rows sequentially (BATCH_SIZE is 1 for now)
                for i in tqdm(range(total_rows)):
                    row = df.iloc[i]

                    # Process this row using LLM
                    try:
                        row_data = (i, row)
                        result = process_row(
                            row_data,
                            client,
                            config,
                            system_prompt,
                            user_prompt_template,
                        )

                        # Update the dataframe with results
                        df.at[i, "is_emoji_proposal"] = result["is_emoji_proposal"]
                        df.at[i, "processing_error"] = result["processing_error"]
                        df.at[i, "document_classification"] = result[
                            "document_classification"
                        ]

                    except Exception as e:
                        df.at[i, "processing_error"] = f"Row processing error: {str(e)}"

                    # Save intermediate results every INTERIM_SAVE_INTERVAL rows
                    if i % INTERIM_SAVE_INTERVAL == 0 and i > 0:
                        interim_file_name = f"utc_register_with_llm_document_classification_interim_{i}.xlsx"
                        interim_path = os.path.join(base_path, interim_file_name)
                        if safe_save_to_excel(df, interim_path):
                            logging.info(f"Saved interim results at row {i}")

                # Save final results
                output_file_name = "utc_register_with_llm_document_classification.xlsx"
                output_path = os.path.join(base_path, output_file_name)
                if safe_save_to_excel(df, output_path):
                    logging.info(f"Processing complete. Results saved successfully.")
                else:
                    logging.error(f"Processing complete but encountered issues saving results.")
