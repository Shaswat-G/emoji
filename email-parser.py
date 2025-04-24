import os
import mailbox
from email.utils import parsedate_to_datetime
import pandas as pd

MBOX_FILE = "test_email_archive.mbox"
ARCHIVE_DIR = "email-archive"
ARCHIVE_PATH = os.path.join(ARCHIVE_DIR, MBOX_FILE)


def ensure_archive_dir():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)


def save_emails_to_file(msgs, archive_path):
    with open(archive_path, "w", encoding="utf-8", errors="replace") as f:
        for msg in msgs:
            f.write(msg.as_string())
            f.write("\n" + "=" * 80 + "\n")  # Separator between emails


def extract_email_features(msg):
    # Standard headers to extract
    headers = [
        "Message-ID",
        "In-Reply-To",
        "References",
        "From",
        "To",
        "Cc",
        "Bcc",
        "Reply-To",
        "Return-Path",
        "Delivered-To",
        "Date",
        "Subject",
        "MIME-Version",
        "Content-Type",
        "Content-Transfer-Encoding",
        "X-Mailer",
        "X-Originating-IP",
        "X-Spam-Status",
        "X-Spam-Score",
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
    # Attachments
    features["has_attachments"] = any(part.get_content_disposition() == "attachment" for part in msg.walk())
    # Email size
    features["size_bytes"] = len(msg.as_bytes())
    # Number of parts
    features["num_parts"] = sum(1 for _ in msg.walk())
    # Content types
    features["content_types"] = ";".join(set(part.get_content_type() for part in msg.walk()))
    
    return features


def find_root(msg, by_id):
    parent = msg.get("In-Reply-To")
    seen = set()
    while parent and parent in by_id and parent not in seen:
        seen.add(parent)
        parent = by_id[parent].get("In-Reply-To")
    return parent if parent in by_id else msg["Message-ID"]


def main():
    # Load mbox
    mbox = mailbox.mbox(MBOX_FILE, factory=None)
    msgs = [msg for msg in mbox]
    by_id = {msg["Message-ID"]: msg for msg in msgs if msg["Message-ID"]}

    # Save all emails to archive
    ensure_archive_dir()
    save_emails_to_file(msgs, ARCHIVE_PATH)

    # Thread roots
    thread_id = {msg["Message-ID"]: find_root(msg, by_id) for msg in msgs if msg["Message-ID"]}

    # Build rows
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
            }
        )
        rows.append(features)

    # DataFrame and export
    df = pd.DataFrame(rows)
    sort_cols = ["thread_id", "date"] if "date" in df.columns else ["thread_id"]
    df.sort_values(sort_cols, inplace=True)
    df.to_excel("email_threads_parsed.xlsx", index=False)
 

if __name__ == "__main__":
    main()
