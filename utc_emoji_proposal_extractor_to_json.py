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

# Extract headers
headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

emoji_list = []
for row in rows[1:]:
    cols = row.find_all(["td", "th"])
    if not cols or len(cols) != len(headers):
        continue
    entry = {headers[i]: cols[i].get_text(strip=True) for i in range(len(headers))}
    emoji_list.append(entry)

# Write to JSON

file_name = "emoji_proposal_table.json"
output_path = os.path.join(base_path, file_name)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(emoji_list, f, ensure_ascii=False, indent=2)
