import pandas as pd
import os
import re


def remove_tz(date_str):
    # Remove trailing timezone abbreviation (e.g., 'CST', 'CDT', etc.)
    return re.sub(r" [A-Z]{3,4}$", "", str(date_str))


if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    COLUMNS = [
        "year",
        "month",
        "date",
        "from_email",
        "from_name",
        "subject",
        "body",
        "thread_id",
        "emoji_relevant",
        "people",
        "emoji_references",
        "entities",
        "summary",
        "other_details",
        "processing_error",
        "api_cost",
        "emoji_chars",
        "unicode_points",
        "emoji_shortcodes",
        "extracted_doc_refs",
    ]

    # Read
    file_name_new = "utc_email_new_with_llm_extraction_doc_refs.xlsx"
    file_path_new = os.path.join(BASE_DIR, file_name_new)
    df_new = pd.read_excel(file_path_new)

    file_name_old = "utc_email_old_with_llm_extraction_doc_refs.xlsx"
    file_path_old = os.path.join(BASE_DIR, file_name_old)
    df_old = pd.read_excel(file_path_old)

    # Process
    df_old["date"] = df_old["date"].apply(remove_tz)
    df_old["date"] = pd.to_datetime(df_old["date"], format="mixed", errors="coerce", utc=True)
    df_old["date"] = df_old["date"].dt.tz_localize(None)

    columns_to_drop_new = [
        "from_name",
        "thread",
        "time",
        "relevance",
        "error",
    ]

    rename_columns = {
        "author": "from_name",
        "message_id": "thread_id",
        "from": "from_email",
    }

    df_old = df_old.drop(columns=columns_to_drop_new, errors="ignore")
    df_old = df_old.rename(columns=rename_columns)
    df_old = df_old[COLUMNS]

    df_new["date"] = pd.to_datetime(df_new["date"], errors="coerce", utc=True)
    df_new["date"] = df_new["date"].dt.tz_localize(None)

    df_new["year"] = df_new["date"].dt.year
    df_new["month"] = df_new["date"].dt.month
    columns_to_drop = [
        "from",
        "has_attachments",
        "size_bytes",
        "thread_subject",
        "mbox_file",
    ]
    df_new = df_new.drop(columns=columns_to_drop, errors="ignore")
    df_new = df_new[COLUMNS]

    # Concatenate
    df_combined = pd.concat([df_new, df_old], ignore_index=True)

    # Save
    output_file_name = "utc_email_combined_with_llm_extraction_doc_ref.xlsx"
    output_file_path = os.path.join(BASE_DIR, output_file_name)
    df_combined.to_excel(output_file_path, index=False)
