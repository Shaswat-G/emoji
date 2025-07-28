# -----------------------------------------------------------------------------
# Script: analyze_accepted_proposal_dataset.py
# Summary: Analyzes accepted emoji proposals, categorizing them into single-concept
#          and combination proposals, and generates summary statistics with
#          document metadata for research analysis.
# Inputs:  emoji_accepted_proposal_dataset.xlsx,
#          utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx
# Outputs: single_concept_accepted_proposals.xlsx, combination_concept_accepted_proposals.xlsx
# Context: Part of emoji proposal research pipeline analyzing UTC's decision-making
#          patterns and proposal categorization for academic study of Unicode
#          emoji standardization processes.
# -----------------------------------------------------------------------------

import pandas as pd
import os
from collections import Counter


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


def main():
    base_dir = os.path.dirname(__file__)

    # Load datasets
    merged = pd.read_excel(
        os.path.join(base_dir, "emoji_accepted_proposal_dataset.xlsx")
    )
    utc_doc_reg = pd.read_excel(
        os.path.join(
            base_dir,
            "utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx",
        )
    )

    # Extract single concept proposals
    single_filter = (merged["accepted_proposal_type"] == "single_concept_proposal") & (
        merged["proposal_doc_num"] != "-"
    )
    single_proposals = set(
        split_and_flatten_proposals(merged[single_filter]["proposal_doc_num"])
    )

    # Process combination proposals
    comb_proposals = split_and_flatten_proposals(
        merged[merged["accepted_proposal_type"] == "combination_proposal"][
            "proposal_doc_num"
        ]
    )

    # Generate counts and merge with metadata
    single_counts = create_proposal_counts(single_proposals)
    comb_counts = create_proposal_counts(comb_proposals, exclude_set=single_proposals)

    single_with_metadata = merge_with_metadata(single_counts, utc_doc_reg)
    comb_with_metadata = merge_with_metadata(comb_counts, utc_doc_reg)

    # Export results
    single_with_metadata.to_excel("single_concept_accepted_proposals.xlsx", index=False)
    comb_with_metadata.to_excel(
        "combination_concept_accepted_proposals.xlsx", index=False
    )


if __name__ == "__main__":
    main()
