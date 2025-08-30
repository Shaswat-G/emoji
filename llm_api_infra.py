# -----------------------------------------------------------------------------
# Script: llm_api_infra.py
# Summary: Minimal OpenAI client wrapper and helpers used by other scripts for
#          calling an LLM with a configuration-driven prompt, parsing outputs
#          and estimating token-based cost. This file centralizes basic API
#          usage patterns so other pipeline scripts can focus on data handling.
# Inputs:  config.yml (or other YAML), API key file path referenced in config,
#          prompt templates and input files
# Outputs: Returns raw LLM content, token usage, and estimated cost to caller
# Context: Designed as a lightweight infra helper to avoid duplicating API call
#          boilerplate across pipeline scripts; not a full-featured production
#          client (no retries, no advanced error handling by default).
# -----------------------------------------------------------------------------


import json
import yaml
import os
import argparse
from openai import OpenAI

def load_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)

def call_api(api_key, config, prompt):
    client = OpenAI(api_key=api_key)

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

    is_json = config["response_format"] == "json_object"
    if is_json:
        try:
            content = json.loads(response.choices[0].message.content.strip())
        except json.JSONDecodeError:
            content = response.choices[0].message.content.strip()
    else:
        content = response.choices[0].message.content.strip()

    tokens = dict(response.usage)
    rates = {'gpt-4o-mini-2024-07-18': {"input": 15e-8, "output": 60e-8},}
    model_rates = rates.get(config["model"])
    cost = tokens['prompt_tokens'] * model_rates["input"] + tokens['completion_tokens'] * model_rates["output"]

    return content, tokens, cost

def main():

    parser = argparse.ArgumentParser(description="LLM API Client")
    parser.add_argument('--config-path', type=str, default='config.yml')
    config = yaml.safe_load(load_file(parser.parse_args().config_path))

    api_key = load_file(config["api_key_path"]).strip()
    prompt_template = load_file(config["prompt_path"])
    input_text = load_file(config["input_path"])

    prompt = prompt_template.format(input_text=input_text)
    content, tokens, cost = call_api(api_key, config, prompt)

    print("\n--- API Response ---")
    if isinstance(content, (dict, list)):
        print(json.dumps(content, indent=2))
    else:
        print(content)

    print(f"\n--- Token Usage: {tokens['total_tokens']} total | Cost: ${cost:.6f} ---")


    output_path = config["output_path"]
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        if isinstance(content, (dict, list)):
            json.dump(content, file, indent=2)
        else:
            file.write(str(content))
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
