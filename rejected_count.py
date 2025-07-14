### This script scrapes a webpage containing rejected emoji proposals, processes the data, and exports it to a CSV file.
# It handles rowspan attributes, extracts relevant information, and normalizes hyphens in the text
# for consistency. The final output is saved as a CSV file named "rejected_proposals.csv".
# Note: There are some rows in Charlotte's page that reference emojis which were accepted in a different form at a later time.
# This script cannot handle those cases and they were included manually.

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

print("Current Working Directory ", os.getcwd())

url = "https://charlottebuff.com/unicode/misc/rejected-emoji-proposals/"

response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.content, "html.parser")

# Print the soup object to inspect its structure
# print(soup.prettify())

tables = soup.find_all("table")
print(len(tables))
print(len(tables[0].find_all("tr")))

table1 = tables[0]
headers = [th.get_text(strip=True) for th in table1.find("thead").find_all("th")]
headers.insert(0, "Category")  # Add category column

data = []
current_category = ""
rows = table1.find("tbody").find_all("tr")
i = 0

while i < len(rows):
    row = rows[i]

    # Check if this is a category header row
    if row.find("th") and row.find("th").get("colspan") == "6":
        current_category = row.find("th").get_text(strip=True)
        i += 1
        continue

    # Get the row's ID if available
    emoji_id = row.get("id", "")
    cells = row.find_all(["td", "th"])

    # Initialize the row data with the category
    row_data = [current_category]

    # Check for rowspan attributes
    rowspan_data = {}
    has_rowspan = False
    for idx, cell in enumerate(cells):
        if cell.has_attr("rowspan"):
            has_rowspan = True
            # Extract cell content, handling links and lists
            if cell.find_all("a"):
                content = []
                for a in cell.find_all("a"):
                    link_text = a.get_text(strip=True)
                    link_href = a.get("href", "")
                    content.append(f"{link_text} ({link_href})")
                cell_content = "; ".join(content)
            elif cell.find("ul"):
                items = [li.get_text(strip=True) for li in cell.find_all("li")]
                cell_content = "; ".join(items)
            else:
                cell_content = cell.get_text(strip=True)

            # Store in rowspan_data
            rowspan_data[idx] = {"content": cell_content, "span": int(cell["rowspan"])}

    # Process the first row
    for idx, cell in enumerate(cells):
        # For cells with rowspan, use stored data
        if idx in rowspan_data:
            row_data.append(rowspan_data[idx]["content"])
        else:
            # Extract content normally
            if cell.find_all("a"):
                content = []
                for a in cell.find_all("a"):
                    link_text = a.get_text(strip=True)
                    link_href = a.get("href", "")
                    content.append(f"{link_text} ({link_href})")
                cell_content = "; ".join(content)
            elif cell.find("ul"):
                items = [li.get_text(strip=True) for li in cell.find_all("li")]
                cell_content = "; ".join(items)
            else:
                cell_content = cell.get_text(strip=True)

            row_data.append(cell_content)

    # Add ID as an extra column for reference
    row_data.append(emoji_id)
    data.append(row_data)

    # If this row has cells with rowspan, process the spanned rows
    if has_rowspan:
        max_rowspan = max(rowspan_data.values(), key=lambda x: x["span"])["span"]

        # Process each spanned row
        for span_idx in range(1, max_rowspan):
            if i + span_idx >= len(rows):
                break

            spanned_row = rows[i + span_idx]
            spanned_id = spanned_row.get("id", "")
            spanned_cells = spanned_row.find_all(["td", "th"])

            # Start with category
            spanned_data = [current_category]

            # Process cells, using rowspan data where needed
            cell_position = 0
            for orig_idx in range(len(cells)):
                if (
                    orig_idx in rowspan_data
                    and span_idx < rowspan_data[orig_idx]["span"]
                ):
                    # Use the rowspan data
                    spanned_data.append(rowspan_data[orig_idx]["content"])
                elif cell_position < len(spanned_cells):
                    # Extract from current spanned row
                    cell = spanned_cells[cell_position]

                    if cell.find_all("a"):
                        content = []
                        for a in cell.find_all("a"):
                            link_text = a.get_text(strip=True)
                            link_href = a.get("href", "")
                            content.append(f"{link_text} ({link_href})")
                        cell_content = "; ".join(content)
                    elif cell.find("ul"):
                        items = [li.get_text(strip=True) for li in cell.find_all("li")]
                        cell_content = "; ".join(items)
                    else:
                        cell_content = cell.get_text(strip=True)

                    spanned_data.append(cell_content)
                    cell_position += 1
                else:
                    # No cell available
                    spanned_data.append("")

            # Add ID and append to data
            spanned_data.append(spanned_id)
            data.append(spanned_data)

        # Skip the spanned rows
        i += max_rowspan
    else:
        # Just advance by one row
        i += 1

# Add 'ID' to headers for the last column
headers.append("ID")


# 1. Standardize column names to lower case with underscores, and rename 'Name (Meaning)' to 'name'
def pep8_col(col):
    col = col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
    if col == "name_meaning":
        return "name"
    return col


headers = [pep8_col(h) for h in headers]

DOC_REF_PATTERN = re.compile(r"(L2/\d{2}[-‐–—−]\d{3})")


def extract_doc_refs(text):
    """Extract document references from the text."""
    if not text or not isinstance(text, str):
        return []
    matches = DOC_REF_PATTERN.findall(text)
    standardized_matches = [re.sub(r"[‐–—−]", "-", match) for match in matches]
    return list(sorted(set(standardized_matches)))


# 2. Build DataFrame with new headers

df = pd.DataFrame(data, columns=headers)


# 3. Clean all cells in the DataFrame by replacing Unicode hyphens and common mis-encodings with ASCII hyphen
def normalize_hyphens(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"(â€“|â€”|â€|‐|–|—|−|‑)", "-", text)
    return text


df = df.map(normalize_hyphens)

# 4. For document and decision columns, extract only the document number (not the link)
doc_cols = [c for c in df.columns if c in ["document", "decision"]]
for col in doc_cols:
    df[col] = df[col].apply(
        lambda x: extract_doc_refs(x)[0] if extract_doc_refs(x) else ""
    )


# 5. Standardize and clean document references in all text columns
def clean_text(text):
    if not isinstance(text, str):
        return text
    # Normalize hyphens
    text = normalize_hyphens(text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


df = df.map(clean_text)

# Remove 'sample' column if it exists
df = df.drop(columns=[col for col in df.columns if col == "sample"], errors="ignore")

df.to_csv("rejected_proposals.csv", index=False)
print("Data exported to rejected_proposals.csv")
