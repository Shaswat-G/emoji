import os
import pandas as pd
import json
import ast
import re
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


base_path = os.getcwd()

emoji_proposal_path = os.path.join(base_path, "emoji_proposal_table.csv")
utc_email_path = os.path.join(
    base_path, "utc_email_combined_with_llm_extraction_doc_ref.xlsx"
)


def safe_literal_eval(val):
    try:
        # Handle empty or NaN values
        if pd.isna(val) or val == "":
            return None
        return ast.literal_eval(val)
    except Exception:
        return val


# Identify columns that need to be parsed as Python objects
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
    email_df = pd.read_excel(utc_email_path)

    # Then apply converters manually to specific columns after loading
    for col in columns_to_eval:
        if col in email_df.columns:
            email_df[col] = email_df[col].apply(safe_literal_eval)
except Exception as e:
    print(f"Error loading or processing the Excel file: {e}")

proposal_df = pd.read_csv(emoji_proposal_path, dtype=str)

# email_df = pd.read_excel(utc_email_path)
# print(proposal_df.sample(3))
# print(email_df.sample(3))


"""
proposal_df contains the following relevant columns:
- doc_num : The proposal number - unique identifier
- proposal_for_1 : The main theme of the proposal identified as a keyword / phrase.
- proposal_for_2 : The secondary theme of the proposal identified as a keyword / phrase - mostly NaN for now.

email_df contains the following relevant columns:
- subject: The subject of the email
- body: The body of the email
- summary: The summary of the email

- doc_ref: A pythonic list of doc_num(s) that the email is referring to.
- emoji_shortcodes: A pythonic list of emoji shortcodes that the email is referring to.

We are trying to find the best match between the proposal_df and email_df based on the doc_num (from proposal_df) and doc_ref (from email_df) and matching the proposal_for_1 and proposal_for_2 with the subject, body, summary and emoji_shortcodes of the email.
For this we need to implement case insentive matching. Also, if there are more than one word in proposal_for_1 or proposal_for_2, we need to check if any of their 2-word of 3-word combinations are present in the subject, body, summary and emoji_shortcodes of the email.
For example, if proposal_for_1 is "Left Pushing Hand" and proposal_for_2 is "Right Pushing Hand", we need to check if "Left Pushing", "Pushing Hand", "Right Pushing", "Pushing Hand" are present in the subject, body, summary and emoji_shortcodes of the email.


Feedback:
1) Restrict search space by year of proposal: Let's say the document number is L2/23-036 - the 4th and 5th character represent the year. Here, the 4th and 5th characteris 23 which means this proposal document is from 2023.
To restrict our search space, let us seach in a 2 year neighbohood of the year of the proposal document. For example, if the proposal document is from 2023, we will search in the emails from 2021 and 2025.
Create a year_neighborhood parameter that takes a positive integer 2 as value, which we can change later if needed. We can filter the emails based on the the "year" column in the dataframe.

2) Include a column to inform me of what was the match based on. For example, if the match was based on doc_num, we can add a column "match_type" which says "doc_num" found in doc_ref. If the match was based on some n-gram of proposal_for_1 or Proposal_for_2, append another string saying found {ngram} in the {field}.
The field should be the corresponding subject/body/summary/emoji_shortcodes.

3) Create a simple confidence score based on the number of matches found. For example, if there are 2 matches found in the subject and 1 match found in the body, we can say that the confidence score is 3. If there are no matches found, we can say that the confidence score is 0. We can add a column "confidence_score" to the final dataframe.

4) Finding a simple string match without suffixed and prefixed spaces will lead to sub-word matching. For example If we are searching for "HARP" is should not match with "Sharp".
5) Retain all the columns from the email_df in the final dataframe. This will help 
"""


def get_ngram_combinations(phrase, ngram_range=(2, 3)):
    """
    Given a phrase, return all n-gram (2-word and 3-word) combinations as a list.
    """
    if not isinstance(phrase, str) or not phrase.strip():
        return []
    words = phrase.strip().split()
    ngrams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)
    return ngrams


def match_phrase_in_text(phrase, text):
    """
    Case-insensitive check if phrase or any of its n-gram combinations are in the text,
    using word boundaries to avoid sub-word matches.
    """
    if not isinstance(text, str):
        return False
    if not isinstance(phrase, str) or not phrase.strip():
        return False
    text = text.lower()
    # Prepare all ngrams (including the phrase itself)
    ngrams = [phrase.strip()] + get_ngram_combinations(phrase)
    for ngram in ngrams:
        ngram = ngram.strip()
        if not ngram:
            continue
        # Use regex for whole word/phrase match (word boundaries)
        pattern = r"\b" + re.escape(ngram.lower()) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def match_proposal_to_email(proposal_row, email_row):
    """
    Returns True if the proposal matches the email by doc_ref or by phrase/combination match.
    """
    # Match by doc_num in doc_ref
    doc_num = str(proposal_row.get("doc_num", "")).strip()
    doc_refs = email_row.get("doc_ref", [])
    if isinstance(doc_refs, str):
        try:
            doc_refs = ast.literal_eval(doc_refs)
        except Exception:
            doc_refs = []
    if doc_num in [str(ref).strip() for ref in doc_refs if ref is not None]:
        return True

    # Prepare fields to search
    fields = []
    for field in ["subject", "body", "summary"]:
        val = email_row.get(field, "")
        if isinstance(val, str):
            fields.append(val)
    # emoji_shortcodes may be a list
    emoji_shortcodes = email_row.get("emoji_shortcodes", [])
    if isinstance(emoji_shortcodes, list):
        fields.extend([str(sc) for sc in emoji_shortcodes if sc])
    elif isinstance(emoji_shortcodes, str):
        try:
            sc_list = ast.literal_eval(emoji_shortcodes)
            if isinstance(sc_list, list):
                fields.extend([str(sc) for sc in sc_list if sc])
        except Exception:
            fields.append(emoji_shortcodes)

    # Check proposal_for_1 and proposal_for_2
    for key in ["proposal_for_1", "proposal_for_2"]:
        phrase = proposal_row.get(key, "")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        for field in fields:
            if match_phrase_in_text(phrase, field):
                return True
    return False


year_neighborhood = 2  # Can be changed as needed


def extract_year_from_docnum(doc_num):
    """
    Extracts the year as integer from doc_num like 'L2/23-036' -> 2023.
    Returns None if not found.
    """
    if not isinstance(doc_num, str) or len(doc_num) < 6:
        return None
    try:
        year_suffix = doc_num[3:5]
        year_int = int(year_suffix)
        # Assume 2000+ for years < 50, else 1900+
        if year_int < 50:
            return 2000 + year_int
        else:
            return 1900 + year_int
    except Exception:
        return None


def match_proposal_to_email_detailed(proposal_row, email_row):
    """
    Returns (confidence_score, match_types) for the proposal/email pair.
    """
    confidence_score = 0
    match_types = []

    # Match by doc_num in doc_ref
    doc_num = str(proposal_row.get("doc_num", "")).strip()
    doc_refs = email_row.get("doc_ref", [])
    if isinstance(doc_refs, str):
        try:
            doc_refs = ast.literal_eval(doc_refs)
        except Exception:
            doc_refs = []
    if doc_num in [str(ref).strip() for ref in doc_refs if ref is not None]:
        confidence_score += 1
        match_types.append("doc_num found in doc_ref")

    # Prepare fields to search
    field_map = {}
    for field in ["subject", "body", "summary"]:
        val = email_row.get(field, "")
        if isinstance(val, str):
            field_map[field] = val
    # emoji_shortcodes may be a list
    emoji_shortcodes = email_row.get("emoji_shortcodes", [])
    if isinstance(emoji_shortcodes, list):
        field_map["emoji_shortcodes"] = " ".join(
            [str(sc) for sc in emoji_shortcodes if sc]
        )
    elif isinstance(emoji_shortcodes, str):
        try:
            sc_list = ast.literal_eval(emoji_shortcodes)
            if isinstance(sc_list, list):
                field_map["emoji_shortcodes"] = " ".join(
                    [str(sc) for sc in sc_list if sc]
                )
            else:
                field_map["emoji_shortcodes"] = emoji_shortcodes
        except Exception:
            field_map["emoji_shortcodes"] = emoji_shortcodes

    # Check proposal_for_1 and proposal_for_2
    for key in ["proposal_for_1", "proposal_for_2"]:
        phrase = proposal_row.get(key, "")
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        ngrams = [phrase.strip()] + get_ngram_combinations(phrase)
        for field, text in field_map.items():
            if not isinstance(text, str):
                continue
            text_lower = text.lower()
            for ngram in ngrams:
                ngram = ngram.strip()
                if not ngram:
                    continue
                pattern = r"\b" + re.escape(ngram.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    confidence_score += 1
                    match_types.append(f'found "{ngram}" in {field}')
    return confidence_score, match_types


# Main matching logic with year filtering and detailed match info
matches = []
email_context_cols = [
    "year",
    "month",
    "date",
    "from_email",
    "from_name",
    "subject",
    "emoji_relevant",
    "people",
    "emoji_references",
    "entities",
    "summary",
    "other_details",
    "emoji_chars",
    "unicode_points",
    "emoji_shortcodes",
    "extracted_doc_refs",
]
for _, proposal_row in proposal_df.iterrows():
    doc_num = proposal_row.get("doc_num", "")
    proposal_year = extract_year_from_docnum(doc_num)
    if proposal_year is None:
        continue
    min_year = proposal_year - year_neighborhood
    max_year = proposal_year + year_neighborhood
    # Filter emails by year column
    filtered_emails = email_df[
        email_df["year"].apply(
            lambda y: min_year <= int(y) <= max_year if pd.notna(y) else False
        )
    ]
    for _, email_row in filtered_emails.iterrows():
        confidence_score, match_types = match_proposal_to_email_detailed(
            proposal_row, email_row
        )
        if confidence_score > 0:
            match_entry = {
                "proposal_doc_num": proposal_row.get("doc_num"),
                "proposal_for_1": proposal_row.get("proposal_for_1"),
                "proposal_for_2": proposal_row.get("proposal_for_2"),
                "match_type": "; ".join(match_types),
                "confidence_score": confidence_score,
            }
            # Add requested email context columns
            for col in email_context_cols:
                match_entry[col] = email_row.get(col)
            matches.append(match_entry)

print(f"Found {len(matches)} matches.")
for match in matches[:5]:
    print(match)

matches_df = pd.DataFrame(matches)
matches_df.to_csv("emoji_proposal_email_matches.csv", index=False)
