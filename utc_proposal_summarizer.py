import os
import pandas as pd
import json
import ast
import re
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict


"""
Goal: To track emoji proposals through time and quantify their time-flow characteristics. For example we would like to measure the time it takes from when the proposal document appears in the UTC docregistry as a row (not a reference) to its last reference in the registry.
We would like to know its velocity of flow through the UTC docregistry and how many times it has been referenced in other documents. We want to capture attention dynamics like who (people) are paying attention to the proposal and how many times it has been referenced in other documents. 
We have to enrich this summarizing data for each proposal with creatively defined relevant metrics with people, entities, emoji counts, etc.
We want to understand how and what are the things that Unicode and UTC members are paying attention to and how does that attention change over time.
We also want to understand year-wise what has been the velocity of processing proposals, time taken, people involved, emojis involved, etc.
DataFrames:
1. emoji_proposal_df: Contains emoji proposal data. Contains columns like 'doc_num', 'proposal_title', 'proposer', 'count' (counts #emojis in proposal).
2. utc_doc_reg_df: Contains UTC document registry - collection all documents whether proposal or not with columns:
- doc_num : document number
- subject : subject of the document
- source : Person who submitted the document
- date : date of the document
- emoji_relevance : relevance of the document to emoji
- extracted_doc_refs : Pythonic list with other doc-references extracted from the document
- emoji_chars : emoji characters found in the document
- unicode_points : unicode points for the emoji characters
- is_emoji_relevant : whether the document is relevant to emoji
- people : Pythonic list of people mentioned in the document
- emoji_references	['Emoji symbols', 'U+1F600', 'Emoji ad-hoc meeting report', 'Emoji: Review of FPDAM8', 'DoCoMo Input on Emoji', 'KDDI Input on Emoji', 'Willcom Input on Emoji']
- entities : Pythonic list of entities mentioned in the document
- summary : Summary of the document.
- description : Description of the document
3. email_match_df: Contains email matches that have matched with corresponding proposals from emoji_proposal_df, Contains columns:
- proposal_doc_num - comes from the emoji_proposal_df - this is the document number.
- proposal_for_1 - comes from the emoji_proposal_df - this is the primary keyword used for matching.
- proposal_for_2 - comes from the emoji_proposal_df - this is the secondary keyword used for matching.
- match_type - describes how the email matches the proposal - which fields triggered the match.
- confidence_score - a score indicating the strength of the match - basically the number of fields that matched.
- year - the year the email was sent.
- month - the month the email was sent.
- date - the date the email was sent.
- from_email - the email address of the sender.
- from_name - the name of the sender.
- subject - the subject line of the email.
- people - Pythonic list of people in that email
"""

base_path = os.getcwd()
emoji_proposal_path = os.path.join(base_path, "emoji_proposal_table.csv")


emoji_proposal_df = pd.read_csv(emoji_proposal_path, dtype=str)
utc_doc_reg_path = os.path.join(base_path, "utc_register_with_llm_extraction.xlsx")
utc_email_path = os.path.join(base_path, "emoji_proposal_email_matches.csv")


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
    utc_doc_reg_df = pd.read_excel(utc_doc_reg_path)

    # Then apply converters manually to specific columns after loading
    for col in columns_to_eval:
        if col in utc_doc_reg_df.columns:
            utc_doc_reg_df[col] = utc_doc_reg_df[col].apply(safe_literal_eval)
except Exception as e:
    print(f"Error loading or processing the Excel file: {e}")


try:
    email_match_df = pd.read_csv(utc_email_path)

    # Then apply converters manually to specific columns after loading
    for col in columns_to_eval:
        if col in email_match_df.columns:
            email_match_df[col] = email_match_df[col].apply(safe_literal_eval)
except Exception as e:
    print(f"Error loading or processing the CSV file: {e}")


def normalize_doc_num(doc_num):
    """
    Normalize document numbers to handle encoding issues and format variations.
    Examples: "L2/23â€'261", "L2/23-261", "l2/23-261" should all match.
    """
    if not isinstance(doc_num, str):
        return ""

    # Extract the standard pattern: L2/YY-XXX
    # Handle various dash characters (hyphen, en-dash, em-dash)
    match = re.search(r"L2/(\d{2})[-\u2013\u2014](\d{3})", doc_num, re.IGNORECASE)
    if match:
        year, number = match.groups()
        return f"L2/{year}-{number}"
    return doc_num


def track_proposal_through_time(proposal_id, utc_df):
    """
    Track a specific proposal through time in the UTC document registry.

    Args:
        proposal_id: The document number to track (e.g., "L2/19-080")
        utc_df: The UTC document register dataframe

    Returns:
        DataFrame containing all mentions of this proposal in chronological order
    """
    # Normalize the proposal ID to handle encoding issues
    normalized_proposal_id = normalize_doc_num(proposal_id)

    # Normalize all doc_nums in the dataframe for matching
    utc_df_copy = utc_df.copy()
    utc_df_copy["normalized_doc_num"] = utc_df_copy["doc_num"].apply(normalize_doc_num)

    # 1. Find the original proposal document (direct match on doc_num)
    direct_matches = utc_df_copy[
        utc_df_copy["normalized_doc_num"] == normalized_proposal_id
    ].copy()

    # 2. Find all documents that reference this proposal
    def references_proposal(refs_list):
        if not isinstance(refs_list, list):
            return False

        # Normalize each reference and check for a match
        for ref in refs_list:
            if normalize_doc_num(ref) == normalized_proposal_id:
                return True
        return False

    reference_matches = utc_df_copy[
        utc_df_copy["extracted_doc_refs"].apply(references_proposal)
    ].copy()

    # 3. Combine direct and reference matches
    all_matches = pd.concat([direct_matches, reference_matches]).drop_duplicates(
        subset=["doc_num"]
    )

    # 4. Mark each document as either the original proposal or a reference
    all_matches["reference_type"] = all_matches["normalized_doc_num"].apply(
        lambda x: "Original Proposal" if x == normalized_proposal_id else "Reference"
    )

    # 5. Sort by date to create a chronological timeline
    if not all_matches.empty and "date" in all_matches.columns:
        try:
            all_matches = all_matches.sort_values("date")
        except Exception as e:
            print(f"Error sorting dates for {proposal_id}: {e}")
            # Fallback: try to convert dates if needed
            if "date" in all_matches.columns:
                all_matches["date"] = pd.to_datetime(
                    all_matches["date"], errors="coerce"
                )
                all_matches = all_matches.sort_values("date")

    return all_matches.drop(columns=["normalized_doc_num"])
