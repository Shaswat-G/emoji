import json
import os

import pandas as pd
import yaml
from openai import OpenAI


def safe_literal_eval(val):
    try:
        # Handle empty or NaN values
        if pd.isna(val) or val == "":
            return None
        return ast.literal_eval(val)
    except Exception:
        return val


def preprocess_prompt_template(template):
    # Escape all curly braces except those for our placeholders
    # Placeholders: proposal_title, summary, description, other_details
    # Replace all { with {{ and } with }}, then un-escape our placeholders
    safe = (
        template.replace("{", "{{").replace("}", "}}")
        .replace("{{proposal_title}}", "{proposal_title}")
        .replace("{{summary}}", "{summary}")
        .replace("{{description}}", "{description}")
        .replace("{{other_details}}", "{other_details}")
    )
    return safe


def create_proposal_prompt(row, prompt_template):
    # Defensive: fallback to empty string if missing
    return prompt_template.format(
        proposal_title=row.get("proposal_title", ""),
        summary=row.get("summary", ""),
        description=row.get("description", ""),
        other_details=row.get("other_details", ""),
    )


def load_config(config_path="proposal_config.yml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_file(file_path, encoding="utf-8"):
    with open(file_path, "r", encoding=encoding) as file:
        return file.read()


def load_api_key(api_key_path):
    with open(api_key_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def call_llm_api(client, config, prompt):
    try:
        response_format = (
            {"type": config["response_format"]}
            if config.get("response_format") == "json_object"
            else None
        )
        messages = [{"role": config.get("role", "user"), "content": prompt}]
        response = client.chat.completions.create(
            model=config["model"],
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1000),
            top_p=config.get("top_p", 1.0),
            response_format=response_format,
            messages=messages,
        )
        if config.get("response_format") == "json_object":
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
            tokens.get("prompt_tokens", 0) * model_rates["input"]
            + tokens.get("completion_tokens", 0) * model_rates["output"]
        )
        return content, tokens, cost, None
    except Exception as e:
        return None, None, 0, str(e)


BASE_DIR = os.getcwd()

file_name = "emoji_proposal_table.csv"
file_path = os.path.join(BASE_DIR, file_name)
proposal_df = pd.read_csv(file_path)

file_name = "utc_register_with_llm_extraction.xlsx"
file_path = os.path.join(BASE_DIR, file_name)

columns_to_eval = [
    "doc_type",
    "extracted_doc_refs",
    "emoji_chars",
    "unicode_points",
    "emoji_keywords_found",
    "emoji_shortcodes",
    "people",
    "emoji_references",
    "entities",
]

try:
    utc_doc_reg_df = pd.read_excel(file_path)

    # Then apply converters manually to specific columns after loading
    for col in columns_to_eval:
        if col in utc_doc_reg_df.columns:
            utc_doc_reg_df[col] = utc_doc_reg_df[col].apply(safe_literal_eval)
except Exception as e:
    print(f"Error loading or processing the Excel file: {e}")


merged_df = pd.merge(left=proposal_df, right=utc_doc_reg_df, how="left", on="doc_num")

# Load config, prompt, and API key
config = load_config("proposal_config.yml")
api_key = load_api_key(config["api_key_path"])
prompt_template_raw = load_file(config.get("prompt_path", "proposal_prompt.txt"))
prompt_template = preprocess_prompt_template(prompt_template_raw)
client = OpenAI(api_key=api_key)

# Prepare output columns
merged_df["llm_keywords"] = None
merged_df["llm_error"] = None
merged_df["llm_api_cost"] = 0.0

for idx, row in merged_df.iterrows():
    prompt = create_proposal_prompt(row, prompt_template)
    content, tokens, cost, error = call_llm_api(client, config, prompt)
    merged_df.at[idx, "llm_api_cost"] = cost
    if error:
        merged_df.at[idx, "llm_error"] = error
        merged_df.at[idx, "llm_keywords"] = None
    else:
        merged_df.at[idx, "llm_error"] = None
        merged_df.at[idx, "llm_keywords"] = content.get("search_keywords", None)

# Optionally save results
merged_df = merged_df[list(proposal_df.columns) + ["llm_keywords"]]
merged_df.to_excel("emoji_proposal_llm_keywords.xlsx", index=False)
