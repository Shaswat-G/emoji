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
from utc_finding_proposals_llm_sweep import (
    load_file,
    load_config,
    load_api_key,
    call_llm_api,
)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":

    config_file_name = "config_for_finding_gender_docs.yml"
    config = load_config(os.path.join(BASE_PATH, config_file_name))
    
    api_key = load_api_key(os.path.join(BASE_PATH, config.get("api_key_file_name", "api_key.txt")))
    user_prompt_template = load_file(os.path.join(BASE_PATH, config.get("user_prompt_file_name")))
    system_prompt = load_file(os.path.join(BASE_PATH, config.get("system_prompt_file_name")))

    client = OpenAI(api_key=api_key)
    
    file_name = "utc_register_with_llm_extraction.xlsx"
    file_path = os.path.join(BASE_PATH, file_name)
    df = pd.read_excel(file_path)
    
    print(client)
    print(df.head())
    
