import os
import yaml
import json
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import re
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


def create_email_prompt(body, thread_subject, prompt_template):
    # Ensure thread_subject and body are strings and not NaN
    if not isinstance(thread_subject, str) or pd.isna(thread_subject):
        thread_subject = ""
    if not isinstance(body, str) or pd.isna(body):
        body = ""
    # Truncate body if needed
    max_chars = 8000
    if len(body) > max_chars:
        body = body[:max_chars] + "... [truncated]"
    return prompt_template.format(body=body, thread_subject=thread_subject)


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
        # Use .get for pandas Series to avoid KeyError
        body = row.get("body", "")
        thread_subject = row.get("subject", "")
        prompt = create_email_prompt(body, thread_subject, prompt_template)
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
    input_file = os.path.join(base_path, "utc_email_old_archive_parsed.xlsx")
    output_file = os.path.join(
        base_path, "utc_email_old_with_llm_extraction_testing.xlsx"
    )
    print(f"Processing {input_file} ...")
    df = pd.read_excel(input_file).sample(5)  # only for testing
    # Add/ensure LLM fields
    for col, default in [
        ("emoji_relevant", False),
        ("people", []),
        ("emoji_references", []),
        ("entities", []),
        ("summary", ""),
        ("description", ""),
        ("other_details", ""),
        ("processing_error", None),
        ("api_cost", 0.0),
    ]:
        if col not in df.columns:
            if isinstance(default, list):
                df[col] = [list(default) for _ in range(len(df))]
            else:
                df[col] = [default] * len(df)
    batch_size = 10
    total_rows = len(df)
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
    # Save final results
    if save_to_excel(df, output_file):
        print(f"Results saved to {output_file}")
    else:
        print(f"Error saving results to {output_file}")
    print(f"Total API cost: ${total_cost:.4f}")
