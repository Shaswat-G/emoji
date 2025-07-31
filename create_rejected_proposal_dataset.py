# -----------------------------------------------------------------------------
# Script: create_rejected_proposal_dataset.py
# Summary: Identifies and extracts rejected emoji proposals from the UTC document register,
#          by removing all proposals present in the accepted proposal datasets (single and combination).
#          Only proposals marked as emoji-related and dated between 2010-2020 are considered.
# Inputs:  utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx,
#          single_concept_accepted_proposals.xlsx,
#          combination_concept_accepted_proposals.xlsx
# Outputs: rejected_proposal_dataset.xlsx
# Context: Part of emoji proposal research pipeline, supporting analysis of rejected
#          proposals for academic study of Unicode emoji standardization processes.
# -----------------------------------------------------------------------------

import pandas as pd
import os


META_COLS = ["doc_num", "subject", "source", "date", "summary", "description"]
PROP_COLS = ["is_emoji_proposal", "is_emoji_mechanism", "is_ideological_argument"]


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
    base_path = os.path.dirname(__file__)

    utc_doc_reg = pd.read_excel(
        os.path.join(
            base_path,
            "utc_register_with_llm_document_classification_and_emoji_proposal_markings.xlsx",
        )
    )

    # Read accepted proposals and build set of accepted doc_nums
    single_accepted = pd.read_excel(
        os.path.join(base_path, "single_concept_accepted_proposals.xlsx")
    )
    comb_accepted = pd.read_excel(
        os.path.join(base_path, "combination_concept_accepted_proposals.xlsx")
    )
    accepted_docnums = set(
        single_accepted["proposal_doc_num"].dropna().astype(str)
    ) | set(comb_accepted["proposal_doc_num"].dropna().astype(str))

    # Filter rows where any of the PROP_COLS is True (if columns are boolean)
    proposals = (
        utc_doc_reg[utc_doc_reg[PROP_COLS].any(axis=1)][META_COLS]
        .drop_duplicates(subset="doc_num")
        .reset_index(drop=True)
    )

    # Filter proposals by year between 2010 and 2020 inclusive
    proposals["year"] = proposals["doc_num"].apply(extract_year_from_docnum)
    proposals = (
        proposals[(proposals["year"] >= 2010) & (proposals["year"] <= 2020)]
        .drop(columns=["year"])
        .reset_index(drop=True)
    )

    # Remove accepted proposals
    rejected_proposals = proposals[
        ~proposals["doc_num"].astype(str).isin(accepted_docnums)
    ].reset_index(drop=True)

    # Save the filtered dataset to an Excel file
    rejected_proposals.to_excel("rejected_proposal_dataset.xlsx", index=False)
    print(f"Rejected proposal dataset created with {len(rejected_proposals)} records")


if __name__ == "__main__":
    main()
