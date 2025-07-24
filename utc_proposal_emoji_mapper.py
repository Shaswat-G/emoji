###############################################################################
# Script: utc_proposal_emoji_mapper.py
# Summary: Extracts emoji-to-proposal mappings, normalized unicode codepoints,
#          and category information from an HTML table. Outputs a structured CSV
#          for downstream analysis and database construction of Unicode emoji proposals.
# Inputs:  emoji_to_proposal_map.html (HTML table)
# Outputs: emoji_to_proposal_map_v2.csv (CSV with columns:
#          emoji_code, unicode_codepoints, unicode_codepoints_list,
#          canonical_codepoints, emoji_name, proposal_doc_num, emoji_category)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data. Designed for robust
#          emoji keying and normalization for database and analysis use.
###############################################################################


from bs4 import BeautifulSoup
import os
import re
import pandas as pd


# For future readers:
# - The HTML table contains columns for emoji code (may be a link), emoji image (img tag), emoji name, and proposal document numbers.
# - This script extracts:
#   1. emoji_code: Unicode code string (from first column)
#   2. unicode_codepoints: Space-separated, normalized (uppercase, U+ prefix) codepoint string for lookup and keying
#   3. unicode_codepoints_list: List of normalized codepoints for programmatic use
#   4. canonical_codepoints: Sorted, space-separated normalized codepoints for canonical matching/searching
#   5. emoji_name: Name of the emoji
#   6. proposal_doc_num: Comma-separated list of proposal document numbers (normalized)
#   7. emoji_category: Category heading (e.g., "2023 - Smileys & Emotion")
#
# - Best practices followed:
#   * All codepoints are normalized to uppercase and use the "U+" prefix
#   * Both string and list representations are provided for flexibility
#   * Canonical (sorted) form is included for robust matching/searching
#   * Output CSV is suitable for database keying and downstream Unicode analysis


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
