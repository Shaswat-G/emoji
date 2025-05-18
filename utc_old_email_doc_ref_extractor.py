import os
import pandas as pd
import re
import emoji
import tqdm


# Compile regex patterns once for better performance
UNICODE_PATTERNS = [
    re.compile(r"[Uu]\+([0-9A-F]{4,6})"),  # U+1F600 format
    re.compile(r"\\u([0-9A-F]{4,6})"),  # \u1F600 format
    re.compile(r"&#x([0-9A-F]{4,6});"),  # &#x1F600; HTML hex entity
    re.compile(r"\\x{([0-9A-F]{4,6})}"),  # \x{1F600} format
    re.compile(r"code[\s-]?point[\s:]+([0-9A-F]{4,6})"),  # "code point 1F600"
]

SHORTCODE_PATTERN = re.compile(r":([a-z0-9_\-+]+):")
DOC_REF_PATTERN = re.compile(r"(L2/\d{2}[-‐–—−]\d{3})")

# write a function to .apply() on the email body that extracts the doc_ref from the email body
def preprocess_text(text):
    # Standardize different types of dashes and remove extra whitespace
    return re.sub(r"\s+", " ", re.sub(r"[‐–—−]", "-", text))


def extract_emojis(text):
    if not text:
        return {"emoji_chars": [], "unicode_points": [], "emoji_shortcodes": []}

    # Extract actual emoji characters
    emoji_list = [emoji_dict["emoji"] for emoji_dict in emoji.emoji_list(text) if isinstance(emoji_dict, dict) and "emoji" in emoji_dict]

    # Demojize to capture shortcodes
    demojized = emoji.demojize(text)

    # Extract Unicode codepoint references
    all_unicode_matches = []
    for pattern in UNICODE_PATTERNS:
        all_unicode_matches.extend(pattern.findall(text))

    # Extract emoji shortcodes
    shortcodes = SHORTCODE_PATTERN.findall(demojized)

    return {
        "emoji_chars": list(set(emoji_list)),
        "unicode_points": list(set(all_unicode_matches)),
        "emoji_shortcodes": list(set(shortcodes)),
    }


def extract_doc_refs(text):
    """Extract document references from the text."""
    if not text:
        return []

    # Find document references and standardize dash types
    matches = DOC_REF_PATTERN.findall(text)
    standardized_matches = [re.sub(r"[‐–—−]", "-", match) for match in matches]

    return list(sorted(set(standardized_matches)))


if __name__ == "__main__":

    BASE_DIR = os.getcwd()
    file_name = "utc_email_old_with_llm_extraction.xlsx"
    file_path = os.path.join(BASE_DIR, file_name)

    df = pd.read_excel(file_path)
    columns_to_drop = ["to", "cc", "bcc", "in_reply_to", "references", "description", "received"]
    df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    df.dropna(subset=["body"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Preprocess the email body text
    df["body"] = df["body"].apply(preprocess_text)

    # Extract emojis and their shortcodes
    emoji_data = df["body"].apply(extract_emojis)
    df = pd.concat([df, emoji_data.apply(pd.Series)], axis=1)

    # Extract document references
    df["doc_ref"] = df["body"].apply(extract_doc_refs)

    # save the results to a new Excel file
    output_file_name = "utc_email_old_with_llm_extraction_doc_refs.xlsx"
    output_file_path = os.path.join(BASE_DIR, output_file_name)
    df.to_excel(output_file_path, index=False)
    print(f"Processed data saved to {output_file_path}")
