# -----------------------------------------------------------------------------
# Script: create_accepted_proposal_dataset.py
# Summary: Creates a unified dataset of accepted emoji proposals by merging
#          emoji chart data with proposal mapping data, categorizing proposals
#          as single-concept or combination types for analysis.
# Inputs:  emoji_chart_with_dates.xlsx, emoji_to_proposal_map_v2.csv
# Outputs: emoji_accepted_proposal_dataset.xlsx
# Context: Foundation dataset creation for emoji proposal research pipeline,
#          establishing ground truth for accepted proposals and their types
#          to support Unicode decision-making pattern analysis.
# -----------------------------------------------------------------------------

import pandas as pd
import os


def clean_proposal_doc_num(s):
    """Sort and clean comma-separated proposal document numbers."""
    if pd.isna(s):
        return s
    return ",".join(sorted(part.strip() for part in str(s).split(",")))


def get_proposal_type(row):
    """Categorize proposals based on data source presence."""
    has_date = pd.notna(row.get("Date"))
    has_proposal = pd.notna(row.get("proposal_doc_num"))

    if has_date and has_proposal:
        return "single_concept_proposal"
    elif has_proposal and not has_date:
        return "combination_proposal"
    return None


def main():
    base_path = os.path.dirname(__file__)

    # Load and process data
    emoji_chart = pd.read_excel(os.path.join(base_path, "emoji_chart_with_dates.xlsx"))
    emoji_map = pd.read_csv(os.path.join(base_path, "emoji_to_proposal_map_v2.csv"))

    # Clean proposal numbers
    emoji_map["proposal_doc_num"] = emoji_map["proposal_doc_num"].apply(clean_proposal_doc_num)

    # Remove unnecessary columns
    emoji_chart.drop(
        columns=["canonical_codepoints", "unicode_codepoints_list", "Name"],
        inplace=True,
        errors="ignore",
    )
    emoji_map.drop(
        columns=["emoji_code", "canonical_codepoints", "unicode_codepoints_list"],
        inplace=True,
        errors="ignore",
    )

    # Merge and categorize
    merged = pd.merge(emoji_map, emoji_chart, how="outer", on="unicode_codepoints")
    merged["accepted_proposal_type"] = merged.apply(get_proposal_type, axis=1)
    merged.sort_values(["proposal_doc_num", "unicode_codepoints"], inplace=True)

    # Export results
    merged.to_excel("emoji_accepted_proposal_dataset.xlsx", index=False)
    print(f"Dataset created with {len(merged)} records")


if __name__ == "__main__":
    main()
