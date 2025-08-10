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
from utc_finding_proposals_llm_sweep import (load_file, load_config, load_api_key, call_llm_api)


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    
    config_file_name = "config.yaml"
    config_file_path = os.path.join(BASE_PATH, config_file_name)
    config = load_config(config_file_path)

    api_key_file_path = os.path.join(BASE_PATH, config.get("api_key_file_name", "api_key.txt"))
    api_key = load_api_key(api_key_file_path)
    
    # client = OpenAI(api_key=api_key)
