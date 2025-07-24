import pandas as pd
from bs4 import BeautifulSoup

# Load the HTML file
with open("emoji_charts.html", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

rows = []
# Find all table rows
for tr in soup.find_all("tr"):
    tds = tr.find_all("td")
    if len(tds) < 4:
        continue  # skip header or malformed rows

    # Extract Date, Source, Count
    date = tds[0].get_text(strip=True)
    source = tds[1].get_text(strip=True)
    count = tds[2].get_text(strip=True)

    # The 4th column contains emojis (possibly multiple)
    emoji_cells = tds[3].find_all("img", title=True)
    for img in emoji_cells:
        import re

        title = img["title"]
        # Use regex to extract all codepoints at the start
        codepoint_matches = list(re.finditer(r"U\+[0-9A-Fa-f]+", title))
        if not codepoint_matches:
            continue  # skip malformed
        # The codepoints are all at the start, possibly separated by spaces
        last_cp_end = codepoint_matches[-1].end()
        codepoint_str = title[:last_cp_end].strip()
        rest = title[last_cp_end:].strip()
        # Normalize codepoints: uppercase, U+ prefix
        codepoints_list = [
            (
                cp.group(0).upper()
                if cp.group(0).startswith("U+")
                else "U+" + cp.group(0).upper()
            )
            for cp in codepoint_matches
        ]
        unicode_codepoints = " ".join(codepoints_list)
        canonical_codepoints = " ".join(sorted(codepoints_list))
        # The next part is the emoji character(s), then the name/description
        if rest:
            rest_parts = rest.split(" ", 1)
            name = rest_parts[1] if len(rest_parts) > 1 else ""
        else:
            name = ""
        rows.append(
            {
                "Date": date,
                "Source": source,
                "Count": count,
                "unicode_codepoints": unicode_codepoints,
                "unicode_codepoints_list": codepoints_list,
                "canonical_codepoints": canonical_codepoints,
                "Name": name,
            }
        )

# Create DataFrame and save as Excel
import unicodedata

# Normalize all string fields to NFC to ensure proper Unicode encoding
for row in rows:
    for k, v in row.items():
        if isinstance(v, str):
            row[k] = unicodedata.normalize("NFC", v)
df = pd.DataFrame(rows)
df.to_excel("emoji_chart_extracted.xlsx", index=False)
print("Extraction complete. Saved to emoji_chart_extracted.xlsx")
