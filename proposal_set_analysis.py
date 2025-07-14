import os
import pandas as pd
import ast


def safe_literal_eval(val):
    """Safely parse string representation of dictionary back to dict"""
    try:
        if pd.isna(val) or val == "":
            return {}
        if isinstance(val, dict):
            return val
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}


def load_set_from_excel(filename, column, filter_func=None):
    """Load a set from Excel file with optional filtering"""
    df = pd.read_excel(os.path.join(os.getcwd(), filename))
    if filter_func:
        df = df[filter_func(df)]
    return set(df[column].dropna())


def load_set_from_csv(filename, column):
    """Load a set from CSV file"""
    df = pd.read_csv(os.path.join(os.getcwd(), filename))
    return set(df[column].dropna())


# Load all three sets
def filter_emoji_proposals(df):
    df["doc_type"] = df["doc_type"].apply(safe_literal_eval)
    proposals_mask = df["doc_type"].apply(
        lambda x: "Proposals" in x if isinstance(x, dict) else False
    )
    return proposals_mask & (df["emoji_relevant"] == True)


all_identified_emoji_proposals = load_set_from_excel("utc_register_with_llm_extraction.xlsx", "doc_num", filter_emoji_proposals)
charlotte_buff_rejected = load_set_from_csv("rejected_proposals.csv", "document")
known_accepted_proposals = load_set_from_csv("emoji_proposal_table.csv", "doc_num")

# Analysis
calculated_rejected = all_identified_emoji_proposals - known_accepted_proposals
overlap = charlotte_buff_rejected.intersection(calculated_rejected)
missing_from_cb = calculated_rejected - charlotte_buff_rejected
extra_in_cb = charlotte_buff_rejected - calculated_rejected
accepted_found = known_accepted_proposals.intersection(all_identified_emoji_proposals)
accepted_missing = known_accepted_proposals - all_identified_emoji_proposals
false_rejects = known_accepted_proposals.intersection(charlotte_buff_rejected)

# Metrics
our_recall = (len(accepted_found) / len(known_accepted_proposals) * 100 if known_accepted_proposals else 0)
cb_recall = (len(overlap) / len(calculated_rejected) * 100 if calculated_rejected else 0)
cb_precision = (len(overlap) / len(charlotte_buff_rejected) * 100 if charlotte_buff_rejected else 0)

# Results
print("EMOJI PROPOSAL SET ANALYSIS")
print("=" * 50)
print(f"All identified emoji proposals: {len(all_identified_emoji_proposals)}")
print(f"Charlotte Buff's rejected list: {len(charlotte_buff_rejected)}")
print(f"Known accepted proposals: {len(known_accepted_proposals)}")
print(f"Calculated rejected proposals: {len(calculated_rejected)}")

print(f"\nVALIDATION:")
print(f"Our recall (accepted found): {our_recall:.1f}%")
print(f"CB recall (rejected found): {cb_recall:.1f}%")
print(f"CB precision: {cb_precision:.1f}%")
print(f"Missing from CB's list: {len(missing_from_cb)}")
print(f"Extra in CB's list: {len(extra_in_cb)}")
print(f"False rejects in CB's list: {len(false_rejects)}")

if missing_from_cb:
    print(f"\nSample missing rejected proposals:")
    for doc in list(missing_from_cb)[:10]:
        print(f"  {doc}")

if false_rejects:
    print(f"\nWARNING - Accepted proposals in rejected list:")
    for doc in list(false_rejects)[:5]:
        print(f"  {doc}")
