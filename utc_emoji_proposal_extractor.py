import json
from bs4 import BeautifulSoup
import os
import re

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


# Read the HTML file
base_path = os.getcwd()
input_file = os.path.join(base_path, "emoji_proposals.html")

with open(input_file, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Find the table (assuming the first table is the one needed)
table = soup.find("table")
rows = table.find_all("tr")

# Define headers explicitly since the table has no <th> header row
headers = ["proposal_link", "proposal_title", "proposer", "count", "emoji_image"]

emoji_list = []
for row in rows:
    cols = row.find_all(["td", "th"])
    if not cols or len(cols) != len(headers):
        continue
    # Extract the proposal link and document number from the first column's <a>
    proposal_link = cols[0].find("a")
    link = proposal_link["href"] if proposal_link else ""
    doc_number_raw = proposal_link.get_text(strip=True) if proposal_link else ""
    # Standardize doc_num using extract_doc_refs
    doc_refs = extract_doc_refs(doc_number_raw)
    doc_number = doc_refs[0] if doc_refs else doc_number_raw
    # Ensure doc_number is normalized (replace any stray non-ASCII hyphens)
    doc_number = re.sub(r"[‐–—−‑]", "-", doc_number)
    # Extract the rest as text, except the last column (image)
    entry = {
        "doc_num": doc_number,
        "proposal_link": link,
        "proposal_title": cols[1].get_text(strip=True),
        "proposer": cols[2].get_text(strip=True),
        "count": cols[3].get_text(strip=True),
        "emoji_image": str(cols[4]),  # Save the HTML for the image cell
    }
    emoji_list.append(entry)

# Write to JSON

file_name = "emoji_proposal_table.json"
output_path = os.path.join(base_path, file_name)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(emoji_list, f, ensure_ascii=False, indent=2)

# Also save as CSV (excluding the Emoji Image column)
import pandas as pd

emoji_proposal_df = pd.DataFrame(emoji_list)
if "emoji_image" in emoji_proposal_df.columns:
    emoji_proposal_df = emoji_proposal_df.drop(columns=["emoji_image"])
emoji_proposal_df.to_csv(
    os.path.join(base_path, "emoji_proposal_table.csv"),
    index=False,
    encoding="utf-8",
)
