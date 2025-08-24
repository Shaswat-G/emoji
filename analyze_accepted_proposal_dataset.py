# -----------------------------------------------------------------------------
# Script: analyze_accepted_proposal_dataset.py
# Summary: Analyzes accepted emoji proposals, categorizing them into single-concept
#          and combination proposals, generates summary statistics with document metadata,
#          and computes the acceptance date for each proposal by finding the earliest date
#          in the dataset after the proposal's original date where the proposal doc number appears.
# Inputs:  emoji_accepted_proposal_dataset.xlsx,
#          utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx
# Outputs: single_concept_accepted_proposals.xlsx (with acceptance_date column),
#          combination_concept_accepted_proposals.xlsx
# Context: Part of emoji proposal research pipeline analyzing UTC's decision-making
#          patterns, proposal categorization, and acceptance timeline for academic study of Unicode
#          emoji standardization processes.
# -----------------------------------------------------------------------------

import pandas as pd
import os
from collections import Counter


META_COLS = ["subject", "source", "date", "summary", "description"]


def split_and_flatten_proposals(series):
    """Extract and flatten proposal document numbers from comma-separated strings."""
    proposals = []
    for val in series.dropna():
        proposals.extend([p.strip() for p in str(val).split(",") if p.strip()])
    return proposals


def create_proposal_counts(proposals, exclude_set=None):
    """Create DataFrame with proposal counts, optionally excluding specified proposals."""
    counter = Counter(proposals)
    if exclude_set:
        for proposal in exclude_set:
            counter.pop(proposal, None)

    return (
        pd.DataFrame(counter.items(), columns=["proposal_doc_num", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def merge_with_metadata(counts_df, metadata_df):
    """Merge proposal counts with UTC document metadata."""
    return (
        pd.merge(
            counts_df,
            metadata_df,
            how="left",
            left_on="proposal_doc_num",
            right_on="doc_num",
        )
        .drop(columns=["doc_num"], errors="ignore")
        .reset_index(drop=True)
    )


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


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load datasets
    merged = pd.read_excel(os.path.join(base_dir, "emoji_accepted_proposal_dataset.xlsx"))
    utc_doc_reg = pd.read_excel(os.path.join(base_dir,"utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx"))
    utc_doc_reg = (utc_doc_reg[["doc_num"] + META_COLS].drop_duplicates(subset="doc_num").reset_index(drop=True))

    # Helper to filter docnums by year range
    def in_year_range(docnum):
        year = extract_year_from_docnum(docnum)
        return year is not None and 2010 <= year <= 2020

    # Extract single concept proposals
    single_filter = (merged["accepted_proposal_type"] == "single_concept_proposal") & (merged["proposal_doc_num"] != "-")
    single_proposals = set(filter(in_year_range, split_and_flatten_proposals(merged[single_filter]["proposal_doc_num"])))

    # Process combination proposals
    comb_proposals = list(filter(in_year_range, split_and_flatten_proposals(merged[merged["accepted_proposal_type"] == "combination_proposal"]["proposal_doc_num"])))

    # Generate counts and merge with metadata
    single_counts = create_proposal_counts(single_proposals)
    comb_counts = create_proposal_counts(comb_proposals, exclude_set=single_proposals)

    single_with_metadata = merge_with_metadata(single_counts, utc_doc_reg)
    comb_with_metadata = merge_with_metadata(comb_counts, utc_doc_reg)

    # Compute acceptance_date for each proposal in single_with_metadata
    merged_dates = merged.copy()
    merged_dates["proposal_doc_num_list"] = merged_dates["proposal_doc_num"].apply(lambda x: [p.strip() for p in str(x).split(",") if p.strip()])
    merged_dates["Date"] = pd.to_datetime(merged_dates["Date"], errors="coerce")
    single_with_metadata["date"] = pd.to_datetime(single_with_metadata["date"], errors="coerce")
    acceptance_dates = []
    for idx, row in single_with_metadata.iterrows():
        docnum = row["proposal_doc_num"]
        base_date = row["date"]
        # Find all rows in merged where docnum is in proposal_doc_num_list
        mask = merged_dates["proposal_doc_num_list"].apply(lambda lst: docnum in lst)
        filtered = merged_dates[mask]
        # Filter for Date > base_date
        filtered = filtered[filtered["Date"] > base_date]
        # Get minimum Date
        if not filtered.empty:
            acceptance_date = filtered["Date"].min()
        else:
            acceptance_date = pd.NaT
        acceptance_dates.append(acceptance_date)
    single_with_metadata["acceptance_date"] = acceptance_dates

    # Tag each single concept proposal with its unique emoji_categories
    def get_emoji_categories(docnum):
        mask = merged["proposal_doc_num"].apply(
            lambda x: docnum in [p.strip() for p in str(x).split(",") if p.strip()]
        )
        return merged.loc[mask, "emoji_category"].unique().tolist()

    single_with_metadata["emoji_categories_list"] = single_with_metadata["proposal_doc_num"].apply(get_emoji_categories)

    # Create binary column is_people_and_body if any category contains 'body'
    def has_body(categories):
        return any("body" in str(cat).lower() for cat in categories)

    single_with_metadata["is_people_and_body"] = single_with_metadata["emoji_categories_list"].apply(has_body)

    # Add processing_time and nature columns
    EXCEPTION_LIMIT_DAYS = 1000  # Customize as needed
    single_with_metadata["processing_time"] = None
    single_with_metadata["nature"] = None
    for idx, row in single_with_metadata.iterrows():
        proposal_date = row["date"]
        acceptance_date = row["acceptance_date"]
        if pd.notnull(proposal_date) and pd.notnull(acceptance_date):
            days = (pd.to_datetime(acceptance_date) - pd.to_datetime(proposal_date)).days
            single_with_metadata.at[idx, "processing_time"] = days
            if days <= 0 or days > EXCEPTION_LIMIT_DAYS:
                single_with_metadata.at[idx, "nature"] = "exception"
            else:
                single_with_metadata.at[idx, "nature"] = "normal"
        else:
            single_with_metadata.at[idx, "processing_time"] = None
            single_with_metadata.at[idx, "nature"] = "exception"

    # Export results
    single_with_metadata.to_excel("single_concept_accepted_proposals.xlsx", index=False)
    comb_with_metadata.to_excel("combination_concept_accepted_proposals.xlsx", index=False)


if __name__ == "__main__":
    main()
