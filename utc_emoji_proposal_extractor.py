import json
from bs4 import BeautifulSoup
import os
import re
import pandas as pd

# DOC_REF_PATTERN and extraction function
DOC_REF_PATTERN = re.compile(r"(L2/\d{2}[-‐–—−]\d{3})")


def extract_doc_refs(text):
    """Extract document references from the text."""
    if not text:
        return []

    # Replace all hyphen-like characters with ASCII hyphen-minus
    normalized_text = re.sub(r"[‐–—−‑]", "-", text)
    matches = DOC_REF_PATTERN.findall(normalized_text)
    # All matches should already have ASCII hyphens, but ensure again
    standardized_matches = [re.sub(r"[‐–—−‑]", "-", match) for match in matches]

    return list(sorted(set(standardized_matches)))


# Improved patterns for extracting the main proposal subject from proposal_title
PROPOSAL_FOR_PATTERNS = [
    # “Proposal for Emoji: X” or “Proposal for Emoji – X”
    re.compile(
        r"(?i)^proposal for emoji[:\s-]\s*(?P<e>.+?)(?:\s+(?:unicode\s+)?emoji)?$",
        re.IGNORECASE,
    ),
    # “X [emoji proposal]”
    re.compile(r"(?i)^(.+?)\s*\[emoji proposal\]$", re.IGNORECASE),
    # “Proposal for X Emoji”
    re.compile(r"(?i)^proposal for\s*(?P<e>.+?)\s+emoji$", re.IGNORECASE),
    # “X Unicode Emoji Proposal”
    re.compile(r"(?i)^(.+?)\s+unicode emoji proposal$", re.IGNORECASE),
    # “X Emoji Proposal”
    re.compile(r"(?i)^(.+?)\s+emoji proposal$", re.IGNORECASE),
    # plain “X Emoji”
    re.compile(r"(?i)^(.+?)\s+emoji$", re.IGNORECASE),
]


def extract_proposal_for(title):
    """Extract the main proposal subject from proposal_title using improved patterns."""
    for pat in PROPOSAL_FOR_PATTERNS:
        m = pat.match(title)
        if m:
            # if we named the group “e”, use it; else use group(1)
            return (m.groupdict().get("e") or m.group(1)).strip()
    # no “emoji” keyword → keep full title
    return title.strip()


def extract_table_from_html(html_file):
    """Extract emoji proposals from a given HTML file."""
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")
    headers = ["proposal_link", "proposal_title", "proposer", "count", "emoji_image"]
    emoji_list = []
    for row in rows:
        cols = row.find_all(["td", "th"])
        if not cols or len(cols) != len(headers):
            continue
        proposal_link = cols[0].find("a")
        link = proposal_link["href"] if proposal_link else ""
        doc_number_raw = proposal_link.get_text(strip=True) if proposal_link else ""
        doc_refs = extract_doc_refs(doc_number_raw)
        doc_number = doc_refs[0] if doc_refs else doc_number_raw
        doc_number = re.sub(r"[‐–—−‑]", "-", doc_number)
        entry = {
            "doc_num": doc_number,
            "proposal_link": link,
            "proposal_title": cols[1].get_text(strip=True),
            "proposer": cols[2].get_text(strip=True),
            "count": cols[3].get_text(strip=True),
            "emoji_image": str(cols[4]),
        }
        # Add proposal_for field
        entry["proposal_for"] = extract_proposal_for(entry["proposal_title"])
        emoji_list.append(entry)
    return emoji_list


# Read both HTML files
base_path = os.getcwd()
input_file_v16 = os.path.join(base_path, "emoji_proposals_v16.html")
input_file_v13 = os.path.join(base_path, "emoji_proposals_v13.html")

emoji_list_v16 = extract_table_from_html(input_file_v16)
emoji_list_v13 = extract_table_from_html(input_file_v13)

# Combine and deduplicate by doc_num (keep first occurrence)
combined = {}
for entry in emoji_list_v16 + emoji_list_v13:
    doc_num = entry["doc_num"]
    if doc_num not in combined:
        combined[doc_num] = entry
emoji_list = list(combined.values())

# Write to JSON
file_name = "emoji_proposal_table.json"
output_path = os.path.join(base_path, file_name)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(emoji_list, f, ensure_ascii=False, indent=2)

# Write to CSV
emoji_proposal_df = pd.DataFrame(emoji_list)

if "emoji_image" in emoji_proposal_df.columns:
    emoji_proposal_df = emoji_proposal_df.drop(columns=["emoji_image"])


emoji_proposal_df.to_csv(
    os.path.join(base_path, "emoji_proposal_table.csv"), index=False, encoding="utf-8"
)
