# -----------------------------------------------------------------------------
# Script: utc_new_email_parser.py
# Summary: Crawls, downloads, parses, and structures Unicode mailing list
#          emails into clean Excel files, extracting metadata, threading, and
#          content for downstream analysis of UTC communications.
# Inputs:  Unicode mailing list archives (.txt) from https://corp.unicode.org/pipermail/unicode/
# Outputs: Excel files per archive in parsed_excels/ with structured email data
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------


import os
import requests
from bs4 import BeautifulSoup
import mailbox
from email.utils import parsedate_to_datetime
import pandas as pd
import re
import numpy as np  # Needed for array type checking

BASE_URL = "https://corp.unicode.org/pipermail/unicode/"
ARCHIVES_DIR = "archives"
PARSED_EXCEL_DIR = "parsed_excels"  # New directory for Excel outputs


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crawl_and_download_archives(limit=None):
    ensure_dir(ARCHIVES_DIR)
    r = requests.get(BASE_URL)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    txt_files = [
        a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".txt")
    ]
    mbox_paths = []
    for txt in txt_files[:limit] if limit else txt_files:
        txt_url = BASE_URL + txt
        mbox_path = os.path.join(ARCHIVES_DIR, txt.replace(".txt", ".mbox"))
        print(f"Fetching {txt_url} → {mbox_path}")
        resp = requests.get(txt_url)
        resp.raise_for_status()
        with open(mbox_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        print(f"  → Saved {mbox_path}")
        mbox_paths.append(mbox_path)
    return mbox_paths


def extract_email_body(msg):
    """Extract plain text body from email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = part.get_content_disposition()
            if ctype == "text/plain" and disp != "attachment":
                try:
                    return part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="replace"
                    )
                except Exception:
                    continue
        return ""
    else:
        try:
            return msg.get_payload(decode=True).decode(
                msg.get_content_charset() or "utf-8", errors="replace"
            )
        except Exception:
            return ""


def parse_from_field(from_field):
    """
    Parse 'From' field of the form 'user at domain (Name)' into email and name.
    Returns (email, name).
    """
    if not from_field:
        return "", ""
    # Try to extract name in parentheses
    name = ""
    m = re.search(r"\(([^)]+)\)", from_field)
    if m:
        name = m.group(1).strip()
    # Remove name part for email extraction
    email_part = re.sub(r"\([^)]+\)", "", from_field).strip()
    # Replace ' at ' or ' AT ' with '@'
    email_part = re.sub(r"\s+at\s+", "@", email_part, flags=re.IGNORECASE)
    # Remove extra spaces
    email_part = email_part.replace(" ", "")
    # Basic email validation
    if "@" not in email_part:
        email_part = ""
    return email_part, name


def extract_email_features(msg):
    # Only keep relevant headers
    headers = [
        "From",
        "Date",
        "Subject",
    ]
    features = {h.lower().replace("-", "_"): msg.get(h) for h in headers}
    # Parse date and remove timezone
    date = features.get("date")
    try:
        date = parsedate_to_datetime(date)
        if date and date.tzinfo is not None:
            date = date.replace(tzinfo=None)
    except Exception:
        pass

    features["date"] = date
    # Parse and sanitize 'From' field
    from_email, from_name = parse_from_field(features.get("from", ""))
    features["from_email"] = from_email
    features["from_name"] = from_name
    # Attachments
    features["has_attachments"] = any(
        part.get_content_disposition() == "attachment" for part in msg.walk()
    )
    # Email size
    features["size_bytes"] = len(msg.as_bytes())
    # Add plain text body
    features["body"] = extract_email_body(msg)

    return features


def remove_illegal_characters(text):
    """Remove characters that are not allowed in Excel worksheets with encoding checks."""
    if not isinstance(text, str):
        # If it's a list or array, convert each element
        if isinstance(text, (list, np.ndarray, pd.Series)):
            return [remove_illegal_characters(str(item)) for item in text]
        return text

    # Try to re-encode as UTF-8 to catch any encoding issues
    try:
        text = text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        text = str(text.encode("ascii", errors="replace").decode("ascii"))

    # Remove control characters and other problematic characters for Excel
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", text)


def find_root(msg, by_id):
    parent = msg.get("In-Reply-To")
    seen = set()
    while parent and parent in by_id and parent not in seen:
        seen.add(parent)
        parent = by_id[parent].get("In-Reply-To")
    return parent if parent in by_id else msg["Message-ID"]


def process_mbox(mbox_path):
    mbox = mailbox.mbox(mbox_path, factory=None)
    msgs = [msg for msg in mbox]
    by_id = {msg["Message-ID"]: msg for msg in msgs if msg["Message-ID"]}
    thread_id = {
        msg["Message-ID"]: find_root(msg, by_id) for msg in msgs if msg["Message-ID"]
    }
    rows = []
    for msg in msgs:
        mid = msg["Message-ID"]
        if not mid:
            continue
        root = thread_id.get(mid, mid)
        root_subject = by_id[root]["Subject"] if root in by_id else msg["Subject"]
        features = extract_email_features(msg)
        features.update(
            {
                "thread_id": root,
                "thread_subject": root_subject,
                "mbox_file": os.path.basename(mbox_path),
            }
        )
        # Sanitize all string fields in features
        for k, v in features.items():
            if isinstance(v, str) or isinstance(v, (list, np.ndarray, pd.Series)):
                features[k] = remove_illegal_characters(v)
        rows.append(features)
    return rows


def main():
    # Step 1: Crawl and download all .txt as .mbox
    mbox_paths = crawl_and_download_archives(10)  # Limit to 10 for testing
    # Step 2: Parse each mbox and save as Excel
    ensure_dir(PARSED_EXCEL_DIR)
    for mbox_path in mbox_paths:
        print(f"Parsing {mbox_path} ...")
        rows = process_mbox(mbox_path)
        df = pd.DataFrame(rows)
        sort_cols = ["thread_id", "date"] if "date" in df.columns else ["thread_id"]
        df.sort_values(sort_cols, inplace=True)
        # Sanitize all string columns in DataFrame before saving
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(remove_illegal_characters)
        excel_name = os.path.splitext(os.path.basename(mbox_path))[0] + ".xlsx"
        excel_path = os.path.join(PARSED_EXCEL_DIR, excel_name)
        df.to_excel(excel_path, index=False)
        print(f"  → Saved {excel_path}")


if __name__ == "__main__":
    main()
