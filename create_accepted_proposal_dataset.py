import pandas as pd
import os

base_path = os.path.dirname(__file__)
emoji_chart_path = os.path.join(base_path, "emoji_chart_with_dates.xlsx")
emoji_to_proposal_map_path = os.path.join(base_path, "emoji_to_proposal_map_v2.csv")


emoji_chart = pd.read_excel(emoji_chart_path)
emoji_to_proposal_map = pd.read_csv(emoji_to_proposal_map_path)


def clean_proposal_doc_num(s):
    if pd.isna(s):
        return s
    parts = [part.strip() for part in str(s).split(",")]
    parts = sorted(parts)
    return ",".join(parts)


emoji_to_proposal_map["proposal_doc_num"] = emoji_to_proposal_map[
    "proposal_doc_num"
].apply(clean_proposal_doc_num)


emoji_chart.drop(
    columns=["canonical_codepoints", "unicode_codepoints_list", "Name"],
    inplace=True,
    errors="ignore",
)
emoji_to_proposal_map.drop(
    columns=["emoji_code", "canonical_codepoints", "unicode_codepoints_list"],
    inplace=True,
    errors="ignore",
)


merged = pd.merge(
    emoji_to_proposal_map,
    emoji_chart,
    how="outer",
    on="unicode_codepoints",
    suffixes=("_chart", "_proposal"),
)
merged.sort_values(by=["proposal_doc_num", "unicode_codepoints"], inplace=True)


# Add accepted_proposal_type column
def get_proposal_type(row):
    # If all columns from both sources are present (not null), it's a single concept proposal
    # We'll check for a column that only exists in emoji_chart (e.g., 'Date')
    # and a column that only exists in emoji_to_proposal_map (e.g., 'proposal_doc_num')
    if pd.notna(row.get("Date")) and pd.notna(row.get("proposal_doc_num")):
        return "single_concept_proposal"
    elif pd.notna(row.get("proposal_doc_num")) and pd.isna(row.get("Date")):
        return "combination_proposal"
    else:
        return None


merged["accepted_proposal_type"] = merged.apply(get_proposal_type, axis=1)

print(merged.head())
print(emoji_to_proposal_map.shape, merged.shape, emoji_chart.shape)
merged.to_excel("emoji_accepted_proposal_dataset.xlsx", index=False)
# print(emoji_chart["unicode_codepoints"].value_counts().head())
# print(emoji_to_proposal_map["unicode_codepoints"].value_counts().head())
# print(merged["unicode_codepoints"].value_counts().head())
