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


def load_set_from_csv(filename, column, filter_func=None):
    """Load a set from CSV file"""
    df = pd.read_csv(os.path.join(os.getcwd(), filename))
    if filter_func:
        df = df[filter_func(df)]
    return set(df[column].dropna())


# --- Filter helpers ---
def in_range(s):
    return (
        isinstance(s, str)
        and len(s) >= 5
        and s[3:5].isdigit()
        and 11 <= int(s[3:5]) <= 20
    )

def filter_utc_doc_reg(df):
    return df["is_emoji_proposal"] == True

def filter_emoji_proposals(df):
    return df["doc_num"].apply(in_range)

def filter_cb_rejections(df):
    return df["document"].apply(in_range)


all_identified_emoji_proposals = load_set_from_excel(
    "utc_register_with_llm_document_classification.xlsx",
    "doc_num",
    filter_utc_doc_reg,
)
charlotte_buff_rejected = load_set_from_csv(
    "rejected_proposals.csv", "document", filter_cb_rejections
)
known_accepted_proposals = load_set_from_csv(
    "emoji_proposal_table.csv", "doc_num", filter_emoji_proposals
)

false_rejects = known_accepted_proposals.intersection(charlotte_buff_rejected)
charlotte_buff_rejected = charlotte_buff_rejected - false_rejects


# Analysis
calculated_rejected = all_identified_emoji_proposals - known_accepted_proposals
overlap = charlotte_buff_rejected.intersection(calculated_rejected)
missing_from_cb = calculated_rejected - charlotte_buff_rejected
extra_in_cb = charlotte_buff_rejected - calculated_rejected
accepted_found = known_accepted_proposals.intersection(all_identified_emoji_proposals)
accepted_missing = known_accepted_proposals - all_identified_emoji_proposals
false_rejects = known_accepted_proposals.intersection(charlotte_buff_rejected)

# Metrics
our_recall = (
    len(accepted_found) / len(known_accepted_proposals) * 100
    if known_accepted_proposals
    else 0
)
cb_recall = len(overlap) / len(calculated_rejected) * 100 if calculated_rejected else 0
cb_precision = (
    len(overlap) / len(charlotte_buff_rejected) * 100 if charlotte_buff_rejected else 0
)

# Results
print("EMOJI PROPOSAL SET ANALYSIS")
print("=" * 50)
print(f"All identified emoji proposals: {len(all_identified_emoji_proposals)}")
print(f"Charlotte Buff's rejected list: {len(charlotte_buff_rejected)}")
print(f"Known accepted proposals: {len(known_accepted_proposals)}")
print(f"Calculated rejected proposals: {len(calculated_rejected)}")

print(f"\nIDENTIFICATION SYSTEM ACCURACY ANALYSIS:")
print("=" * 50)
print(f"Known accepted proposals found by our system: {len(accepted_found)}")
print(f"Known accepted proposals missed by our system: {len(accepted_missing)}")
print(f"Our system recall on accepted proposals: {our_recall:.1f}%")

if len(all_identified_emoji_proposals) > 0:
    acceptance_rate_identified = (
        len(accepted_found) / len(all_identified_emoji_proposals) * 100
    )
    print(f"Acceptance rate in our identified set: {acceptance_rate_identified:.1f}%")

if len(known_accepted_proposals) > 0:
    total_acceptance_rate = (
        len(known_accepted_proposals)
        / (len(known_accepted_proposals) + len(charlotte_buff_rejected))
        * 100
        if charlotte_buff_rejected
        else 100
    )
    print(f"Overall acceptance rate (known data): {total_acceptance_rate:.1f}%")

# Analysis of what we're missing
if accepted_missing:
    print(f"\nMISSED ACCEPTED PROPOSALS ANALYSIS:")
    print(f"We missed {len(accepted_missing)} accepted proposals")
    print(f"Sample missed accepted proposals:")
    for doc in list(accepted_missing)[:10]:
        print(f"  {doc}")

# Analysis of identification quality
extra_identified = (
    all_identified_emoji_proposals - known_accepted_proposals - charlotte_buff_rejected
)
print(f"\nIDENTIFICATION QUALITY:")
print(f"Novel proposals identified (not in any known list): {len(extra_identified)}")
if extra_identified:
    print(f"Sample novel identifications:")
    for doc in list(extra_identified)[:10]:
        print(f"  {doc}")

# Completeness estimation
if our_recall > 0:
    estimated_total_accepted = len(known_accepted_proposals) / (our_recall / 100)
    estimated_missing_accepted = estimated_total_accepted - len(
        known_accepted_proposals
    )
    print(f"\nCOMPLETENESS ESTIMATION:")
    print(f"Estimated total accepted proposals: {estimated_total_accepted:.0f}")
    print(f"Estimated missing from accepted list: {estimated_missing_accepted:.0f}")

print(f"\nVALIDATION:")
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
