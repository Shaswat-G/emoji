# -----------------------------------------------------------------------------
# Script: utc_proposal_emoji_mapper.py
# Summary: Extracts emoji-to-proposal mappings and category information from an
#          HTML table, outputting a structured CSV for downstream analysis of
#          Unicode emoji proposal processes.
# Inputs:  emoji_to_proposal_map.html (HTML table)
# Outputs: emoji_to_proposal_map.csv (CSV with emoji_code, emoji_name,
#          proposal_doc_num, emoji_category)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------


from bs4 import BeautifulSoup
import os
import re
import pandas as pd


# For future readers:
# The table contains 3 columns:
# 1. emoji_code, (it is also a link, we do not care about the link, we do not need it),
# 2. emoji_image, (we do not need this, can be dropped)
# 3. emoji_name,
# 4. proposal_doc_num (this should be extracted in the form of a pythonic list -> comma-separated list.)
# The issue is, sometimes, some rows just have a single heading like "2023 — Smileys & Emotion", "2023 — People & Body".
# We can extract it into a separate column as all the rows following such a row belong to this category - should be called emoji_category
# Hence your output should be 1 csv file with 4 columns : emoji_code, emoji_name, proposal_doc_num, emoji_category.


def parse_emoji_proposal_map(html_path, output_csv):
    from bs4 import BeautifulSoup
    import pandas as pd

    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    table = soup.find("table")
    rows = table.find_all("tr")

    data = []
    current_category = None

    for row in rows:
        # Check for category row: single <th> with colspan=4
        ths = row.find_all("th")
        if len(ths) == 1 and ths[0].has_attr("colspan") and ths[0]["colspan"] == "4":
            # Extract category text (may be inside <a>)
            category_text = ths[0].get_text(strip=True)
            # Sanitize emoji_category: replace all hyphen-like characters with ASCII hyphen-minus
            category_text = re.sub(r"[‐–—−‑]", "-", category_text)
            current_category = category_text
            continue
        cols = row.find_all("td")
        # Data row: must have at least 4 columns (code, image, name, proposal)
        if len(cols) >= 4:
            # 1st col: emoji code (may be a link)
            code_cell = cols[0]
            emoji_code = code_cell.get_text(strip=True)
            # 2nd col: emoji image (extract unicode codepoints from title)
            image_cell = cols[1]
            img_tag = image_cell.find("img")
            codepoints_list = []
            if img_tag and img_tag.has_attr("title"):
                title_text = img_tag["title"]
                # Extract all codepoints (single or multiple) from title
                codepoints = re.findall(r"U\+[0-9A-Fa-f]{4,}", title_text)
                # Normalize: uppercase, ensure U+ prefix
                codepoints_list = [
                    cp.upper() if cp.startswith("U+") else "U+" + cp.upper()
                    for cp in codepoints
                ]
            # Space-separated string for lookup
            unicode_codepoints = " ".join(codepoints_list)
            # Canonical sorted form (for matching/searching)
            canonical_codepoints = " ".join(sorted(codepoints_list))
            # 3rd col: emoji name
            emoji_name = cols[2].get_text(strip=True)
            # 4th col: proposal doc num (may be comma-separated)
            proposal_cell = cols[3]
            proposal_doc_num = proposal_cell.get_text(strip=True)
            # Sanitize proposal_doc_num: replace all hyphen-like characters with ASCII hyphen-minus
            proposal_doc_num = re.sub(r"[‐–—−‑]", "-", proposal_doc_num)
            # Convert to comma-separated list (as string)
            proposal_doc_num_list = [
                x.strip() for x in proposal_doc_num.split(",") if x.strip()
            ]
            proposal_doc_num_pylist = ",".join(proposal_doc_num_list)
            data.append(
                {
                    "emoji_code": emoji_code,
                    "unicode_codepoints": unicode_codepoints,
                    "unicode_codepoints_list": codepoints_list,
                    "canonical_codepoints": canonical_codepoints,
                    "emoji_name": emoji_name,
                    "proposal_doc_num": proposal_doc_num_pylist,
                    "emoji_category": current_category,
                }
            )
    # Write to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    html_path = os.path.join(os.path.dirname(__file__), "emoji_to_proposal_map.html")
    output_csv = os.path.join(os.path.dirname(__file__), "emoji_to_proposal_map_v2.csv")
    parse_emoji_proposal_map(html_path, output_csv)
