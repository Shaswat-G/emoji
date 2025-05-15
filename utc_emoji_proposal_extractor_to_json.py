import json
from bs4 import BeautifulSoup
import os

# Read the HTML file
base_path = os.getcwd()
input_file = os.path.join(base_path, "emoji_proposals.html")

with open(input_file, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

# Find the table (assuming the first table is the one needed)
table = soup.find("table")
rows = table.find_all("tr")

# Define headers explicitly since the table has no <th> header row
headers = ["Proposal Link", "Proposal Title", "Proposer(s)", "Count", "Emoji Image"]

emoji_list = []
for row in rows:
    cols = row.find_all(["td", "th"])
    if not cols or len(cols) != len(headers):
        continue
    # Extract the proposal link and document number from the first column's <a>
    proposal_link = cols[0].find("a")
    link = proposal_link["href"] if proposal_link else ""
    doc_number = proposal_link.get_text(strip=True) if proposal_link else ""
    # Extract the rest as text, except the last column (image)
    entry = {
        "Document Number": doc_number,
        headers[0]: link,
        headers[1]: cols[1].get_text(strip=True),
        headers[2]: cols[2].get_text(strip=True),
        headers[3]: cols[3].get_text(strip=True),
        headers[4]: str(cols[4]),  # Save the HTML for the image cell
    }
    emoji_list.append(entry)

# Write to JSON

file_name = "emoji_proposal_table.json"
output_path = os.path.join(base_path, file_name)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(emoji_list, f, ensure_ascii=False, indent=2)
