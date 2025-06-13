# -----------------------------------------------------------------------------
# Script: utc_new_email_llmsweep.py
# Summary: Sweeps parsed Unicode emails with an LLM to extract structured
#          metadata and summaries, saving results to new Excel files for
#          downstream analysis of UTC communications.
# Inputs:  Parsed email Excel files (parsed_excels/), email_config.yml,
#          email_prompt.txt, OpenAI API key
# Outputs: Excel files with LLM-extracted fields (parsed_excels/*_llm.xlsx)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------


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


def load_file(file_path, encoding="utf-8"):
    try:
        with open(file_path, "r", encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)


def load_config(config_path="email_config.yml"):
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit(1)


def load_api_key(api_key_path):
    try:
        with open(api_key_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file '{api_key_path}' not found.")
        exit(1)


def call_llm_api(client, config, prompt):
    try:
        response_format = (
            {"type": config["response_format"]}
            if config["response_format"] == "json_object"
            else None
        )
        messages = [{"role": config["role"], "content": prompt}]
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
        if config["response_format"] == "json_object":
            try:
                content = json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                content = response.choices[0].message.content.strip()
        else:
            content = response.choices[0].message.content.strip()
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


def remove_illegal_characters(text):
    if not isinstance(text, str):
        if isinstance(text, (list, np.ndarray, pd.Series)):
            return [remove_illegal_characters(str(item)) for item in text]
        return text
    try:
        text = text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        text = str(text.encode("ascii", errors="replace").decode("ascii"))
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def create_email_prompt(subject, body, thread_subject, prompt_template):
    # Truncate body if needed
    max_chars = 8000
    if len(body) > max_chars:
        body = body[:max_chars] + "... [truncated]"
    return prompt_template.format(
        subject=subject, body=body, thread_subject=thread_subject
    )


def process_row(row_data, client, config, prompt_template):
    idx, row = row_data
    result = {
        "idx": idx,
        "emoji_relevant": False,
        "people": [],
        "emoji_references": [],
        "entities": [],
        "summary": "",
        "description": "",
        "other_details": "",
        "processing_error": None,
        "api_cost": 0,
    }
    try:
        subject = row.get("subject", "")
        body = row.get("body", "")
        thread_subject = row.get("thread_subject", "")
        prompt = create_email_prompt(subject, body, thread_subject, prompt_template)
        content, tokens, cost, error = call_llm_api(client, config, prompt)
        if error:
            result["processing_error"] = f"API error: {error}"
        elif content:
            if isinstance(content, dict):
                result["emoji_relevant"] = content.get("emoji_relevant", False)
                result["people"] = content.get("people", [])
                result["emoji_references"] = content.get("emoji_references", [])
                result["entities"] = content.get("entities", [])
                result["summary"] = content.get("summary", "")
                result["description"] = content.get("description", "")
                result["other_details"] = content.get("other_details", "")
            else:
                result["processing_error"] = (
                    "Response format error: Expected JSON object"
                )
            result["api_cost"] = cost
        else:
            result["processing_error"] = "No content returned from API"
    except Exception as e:
        result["processing_error"] = f"Processing error: {str(e)}"
    return result


def save_to_excel(df, file_path):
    try:
        save_df = df.copy()
        text_columns = ["summary", "description", "other_details", "processing_error"]
        for col in text_columns:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(
                    lambda x: remove_illegal_characters(x) if x is not None else None
                )
        list_columns = ["people", "emoji_references", "entities"]
        for col in list_columns:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(
                    lambda x: (
                        str(remove_illegal_characters(x)) if x is not None else None
                    )
                )
        save_df.to_excel(file_path, index=False, engine="openpyxl")
        print(f"Saved: {file_path}")
        return True
    except Exception as e:
        print(f"Error saving to Excel: {str(e)}")
        return False


if __name__ == "__main__":
    multiprocessing.freeze_support()
    config = load_config("email_config.yml")
    api_key = load_api_key(config["api_key_path"])
    prompt_template = load_file(config.get("prompt_path", "email_prompt.txt"))
    client = OpenAI(api_key=api_key)
    base_path = os.getcwd()
    parsed_excel_dir = os.path.join(base_path, "parsed_excels")
    excel_files = [f for f in os.listdir(parsed_excel_dir) if f.endswith(".xlsx")]
    for excel_file in excel_files:
        file_path = os.path.join(parsed_excel_dir, excel_file)
        print(f"Processing {file_path} ...")
        df = pd.read_excel(file_path)
        # Add/ensure LLM fields
        for col, default in [
            ("emoji_relevant", False),
            ("people", None),
            ("emoji_references", None),
            ("entities", None),
            ("summary", None),
            ("description", None),
            ("other_details", None),
            ("processing_error", None),
            ("api_cost", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default
        batch_size = 10
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size
        total_cost = 0.0
        for i in tqdm(range(0, total_rows, batch_size)):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            row_data = [(idx, row) for idx, row in batch_df.iterrows()]
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(process_row, rd, client, config, prompt_template)
                    for rd in row_data
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"A worker thread failed: {str(e)}")
            batch_cost = 0.0
            for result in results:
                idx = result["idx"]
                df.at[idx, "emoji_relevant"] = result["emoji_relevant"]
                df.at[idx, "people"] = result["people"]
                df.at[idx, "emoji_references"] = result["emoji_references"]
                df.at[idx, "entities"] = result["entities"]
                df.at[idx, "summary"] = result["summary"]
                df.at[idx, "description"] = result["description"]
                df.at[idx, "other_details"] = result["other_details"]
                df.at[idx, "processing_error"] = result["processing_error"]
                df.at[idx, "api_cost"] = result["api_cost"]
                batch_cost += result["api_cost"]
            total_cost += batch_cost
            print(f"Batch cost: ${batch_cost:.4f} | Running total: ${total_cost:.4f}")
            # Save interim every 3 batches
            if (i // batch_size) % 3 == 0 and i > 0:
                interim_file_name = (
                    f"{os.path.splitext(excel_file)[0]}_llm_interim_{i}.xlsx"
                )
                interim_path = os.path.join(parsed_excel_dir, interim_file_name)
                save_to_excel(df, interim_path)
        # Save final results
        output_file_name = f"{os.path.splitext(excel_file)[0]}_llm.xlsx"
        output_path = os.path.join(parsed_excel_dir, output_file_name)
        if save_to_excel(df, output_path):
            print(f"Results saved for {excel_file}")
        else:
            print(f"Error saving results for {excel_file}")
        print(f"Total API cost for {excel_file}: ${total_cost:.4f}")
