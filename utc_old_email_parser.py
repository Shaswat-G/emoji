# -----------------------------------------------------------------------------
# Script: utc_old_email_parser.py
# Summary: Fetches, parses, and structures old-format Unicode mailing list
#          emails into a clean Excel file for downstream analysis.
# Inputs:  mail_archive_old_format.xlsx (email index), Unicode mailing list
#          archive (HTML)
# Outputs: utc_email_old_archive_parsed.xlsx (parsed and cleaned emails)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re   
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata

# Setup logging
log_path = os.path.join(os.getcwd(), "email_extraction.log")
logging.basicConfig(
    filename=log_path,
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
)


class EmailExtractor:
    def __init__(self, base_url):
        self.base_url = base_url

    @staticmethod
    def sanitize_text(text):
        """Clean text to avoid Unicode encoding issues and illegal Excel characters."""
        if text is None or not isinstance(text, str):
            return text
        try:
            # Normalize Unicode (NFC form tends to work best)
            text = unicodedata.normalize("NFC", text)
            # Remove illegal Unicode characters for Excel/openpyxl
            # Remove control chars, surrogates, noncharacters
            illegal_ranges = [
                (0x00, 0x08),
                (0x0B, 0x0C),
                (0x0E, 0x1F),
                (0x7F, 0x9F),
                (0xD800, 0xDFFF),  # Surrogates
                (0xFFFE, 0xFFFF),
            ]

            def is_illegal(char):
                cp = ord(char)
                for start, end in illegal_ranges:
                    if start <= cp <= end:
                        return True
                return False

            text = "".join(c for c in text if not is_illegal(c))
            # Replace any remaining problematic bytes
            text = text.encode("utf-8", errors="replace").decode("utf-8")
            return text
        except Exception:
            return "[Encoding Error]"

    @staticmethod
    def clean_dataframe_strings(df):
        """Sanitize all string columns in the DataFrame to avoid Excel errors."""
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: EmailExtractor.sanitize_text(x) if pd.notna(x) else x
                )
        return df

    # --- Utility methods for parsing ---
    @staticmethod
    def get_meta_content(soup, name):
        meta = soup.find("meta", {"name": name})
        return meta["content"] if meta else ""

    @staticmethod
    def extract_from_mailto(link):
        """Extract email address before '?' from a mailto link."""
        if not link:
            return ""
        match = re.search(r"mailto:([^?]+)", link)
        return match.group(1) if match else link.split("?")[0]

    @staticmethod
    def parse_html_comments(html):
        """Return all HTML comments as a list of strings."""
        return re.findall(r"<!--\s*([^>]+?)\s*-->", html)

    @staticmethod
    def parse_headers_from_comments(comments):
        headers = {}
        for comment in comments:
            matches = re.findall(r'([a-zA-Z0-9_-]+)="([^"]*)"', comment)
            for k, v in matches:
                headers[k.lower()] = v
        return headers

    @staticmethod
    def group_quoted_blocks(lines):
        """Group consecutive quoted lines (even if separated by blank lines) into quote blocks."""
        reformatted_lines = []
        prev_quote_level = 0
        quote_buffer = []
        for line in lines:
            stripped = line.lstrip()
            quote_level = 0
            while quote_level < len(stripped) and stripped[quote_level] == ">":
                quote_level += 1
            is_blank = stripped == ""
            if quote_level > 0 or (is_blank and prev_quote_level > 0 and quote_buffer):
                if quote_level > 0 or is_blank:
                    quote_buffer.append(line)
                    prev_quote_level = (
                        quote_level if quote_level > 0 else prev_quote_level
                    )
                    continue
            if prev_quote_level > 0 and quote_buffer:
                reformatted_lines.append(f"[Quote level {prev_quote_level}]")
                reformatted_lines.extend(quote_buffer)
                reformatted_lines.append("[End quote]")
                quote_buffer = []
            reformatted_lines.append(line)
            prev_quote_level = 0
        if prev_quote_level > 0 and quote_buffer:
            reformatted_lines.append(f"[Quote level {prev_quote_level}]")
            reformatted_lines.extend(quote_buffer)
            reformatted_lines.append("[End quote]")
        return "\n".join(reformatted_lines).strip()

    def extract_email_fields(self, soup, html, is_hypermail_215=False):
        """Extract all email fields from soup and html, for both 2.2.0 and 2.1.5."""
        # --- Meta and headers ---
        title = soup.title.text.strip() if soup.title else ""
        subject = self.get_meta_content(soup, "Subject")
        author = self.get_meta_content(soup, "Author")
        date_str = self.get_meta_content(soup, "Date")

        # --- HTML comments and headers ---
        comments = self.parse_html_comments(html)
        headers = self.parse_headers_from_comments(comments)

        # --- Message-ID, References ---
        message_id = headers.get("id", "")
        references = headers.get("references", "")

        # --- From, To, Cc, Bcc, Date ---
        from_addr, from_name, to_addr, cc_addr, bcc_addr, date = "", "", "", "", "", ""
        if is_hypermail_215:
            # 2.1.5: extract from <p> block
            p_blocks = soup.find_all("p")
            if p_blocks:
                p_html = str(p_blocks[0])
                from_match = re.search(
                    r'<strong>From:</strong>\s*(.*?)\(<a href="mailto:([^"]+)"',
                    p_html,
                    re.DOTALL,
                )
                if from_match:
                    from_name = from_match.group(1).strip()
                    from_addr = from_match.group(2).strip().split("?")[0]
                else:
                    email_match = re.search(r'<a href="mailto:([^"]+)"', p_html)
                    if email_match:
                        from_addr = email_match.group(1).strip().split("?")[0]
                date_match = re.search(r"<strong>Date:</strong>\s*([^<]+)", p_html)
                date = date_match.group(1).strip() if date_match else date_str
        else:
            # 2.2.0: extract from <address class="headers">
            mail_div = soup.find("div", {"class": "mail"})
            if mail_div:
                from_span = mail_div.find("span", {"id": "from"})
                if from_span:
                    email_link = from_span.find("a")
                    if email_link:
                        from_addr = self.extract_from_mailto(
                            email_link.get("href", "")
                        ).replace("_at_", "@")
                    else:
                        from_addr = from_span.get_text(" ", strip=True).replace(
                            "_at_", "@"
                        )
                    from_text = from_span.get_text(" ", strip=True)
                    name_match = re.match(r"From\s*:\s*(.*?)\s*<", from_text)
                    if name_match:
                        from_name = name_match.group(1).strip()
                address_block = mail_div.find("address", {"class": "headers"})
                if address_block:
                    for line in address_block.stripped_strings:
                        if re.match(r"To\s*: (.+)", line):
                            to_addr = re.match(r"To\s*: (.+)", line).group(1)
                        elif re.match(r"Cc\s*: (.+)", line):
                            cc_addr = re.match(r"Cc\s*: (.+)", line).group(1)
                        elif re.match(r"Bcc\s*: (.+)", line):
                            bcc_addr = re.match(r"Bcc\s*: (.+)", line).group(1)
                # fallback from headers
                if not to_addr:
                    to_addr = headers.get("to", "")
                if not cc_addr:
                    cc_addr = headers.get("cc", "")
                if not bcc_addr:
                    bcc_addr = headers.get("bcc", "")
                # Date
                date_span = mail_div.find("span", {"id": "date"})
                date_value = (
                    date_span.find(string=True, recursive=False).strip()
                    if date_span
                    else ""
                )
                date_value = date_value.lstrip(":").strip()
                date = (
                    date_value
                    or headers.get("sent", "")
                    or headers.get("date", "")
                    or ""
                )

        # --- Body extraction ---
        body = ""
        if is_hypermail_215:
            body_match = re.search(
                r'<!-- body="start" -->(.*?)<!-- body="end" -->',
                html,
                re.DOTALL | re.IGNORECASE,
            )
            if body_match:
                body_html = body_match.group(1)
                body_html = re.sub(r"<br\s*/?>", "\n", body_html)
                body_html = re.sub(r"<[^>]+>", "", body_html)
                body_html = re.sub(r"&nbsp;", " ", body_html)
                body_html = re.sub(r"&lt;", "<", body_html)
                body_html = re.sub(r"&gt;", ">", body_html)
                body_html = re.sub(r"&amp;", "&", body_html)
                body_lines = [line.strip() for line in body_html.strip().splitlines()]
                body = self.group_quoted_blocks(body_lines)
        else:
            mail_div = soup.find("div", {"class": "mail"})
            message_body = ""
            if mail_div:
                body_p = mail_div.find("p")
                if body_p:
                    for element in body_p.contents:
                        if getattr(element, "name", None) == "br":
                            message_body += "\n"
                        else:
                            message_body += str(element).strip()
                    message_body = re.sub(r"<!--nospam-->", "", message_body)
                    message_body = re.sub(r"<[^>]+>", "", message_body)
            if message_body:
                lines = [line.rstrip() for line in message_body.splitlines()]
                body = self.group_quoted_blocks(lines)

        # --- Compose result ---
        sanitize = self.sanitize_text
        return {
            "subject": sanitize((subject or title).strip()),
            "from": sanitize(from_addr.replace("_at_", "@")),
            "from_name": sanitize(from_name.strip()),
            "to": sanitize(to_addr.strip()),
            "cc": sanitize(cc_addr.strip()),
            "bcc": sanitize(bcc_addr.strip()),
            "date": sanitize(date.strip()),
            "received": "",  # can be added if needed
            "message_id": sanitize(message_id.strip()),
            "in_reply_to": "",  # skip thread hash
            "references": sanitize(references.strip()),
            "body": sanitize(body.strip()),
        }

    def fetch_and_parse_email(self, row):
        month_str = f"m{row['month']:02d}"
        archive_suffix = f"y{row['year']}-{month_str}"
        suffix_url = f"{self.base_url}{archive_suffix}/"
        email_url = f"{suffix_url}{row['thread']}"

        empty_fields = {
            "subject": "",
            "from": "",
            "from_name": "",
            "to": "",
            "cc": "",
            "bcc": "",
            "date": "",
            "received": "",
            "message_id": "",
            "in_reply_to": "",
            "references": "",
            "body": "",
        }

        try:
            response = requests.get(email_url, timeout=30)
            if (
                response.status_code == 403
                or b"You don't have permission to access this resource"
                in response.content
            ):
                error_msg = (
                    "Forbidden: You don't have permission to access this resource."
                )
                logging.error(f"Forbidden for URL: {email_url} | Reason: {error_msg}")
                return {**row.to_dict(), **empty_fields, "error": error_msg}
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            html = str(soup)
            is_2011 = int(row["year"]) == 2011
            meta_generator = soup.find("meta", {"name": "generator"})
            generator_content = meta_generator["content"] if meta_generator else ""
            is_hypermail_215 = (
                is_2011 and "hypermail 2.1.5" in generator_content.lower()
            )
            try:
                email_message = self.extract_email_fields(soup, html, is_hypermail_215)
                if email_message:
                    return {**row.to_dict(), **email_message, "error": ""}
                else:
                    error_msg = "Email extraction failed"
                    logging.error(
                        f"Extraction failed for URL: {email_url} | Reason: {error_msg}"
                    )
                    return {
                        **row.to_dict(),
                        **empty_fields,
                        "error": error_msg,
                    }
            except Exception as e:
                logging.error(
                    f"Extraction error for URL: {email_url} | Error: {str(e)}"
                )
                return {**row.to_dict(), **empty_fields, "error": str(e)}
        except Exception as e:
            logging.error(f"Request error for URL: {email_url} | Error: {str(e)}")
            return {**row.to_dict(), **empty_fields, "error": str(e)}

    def process_archive(self, email_archive, max_workers=8):
        expanded_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.fetch_and_parse_email, row)
                for _, row in email_archive.iterrows()
            ]
            for future in as_completed(futures):
                try:
                    expanded_rows.append(future.result())
                except Exception as e:
                    logging.error(f"Unexpected error in worker: {str(e)}")
        return pd.DataFrame(expanded_rows)


def main():

    base_path = os.getcwd()
    email_archive_path = os.path.join(base_path, "mail_archive_old_format.xlsx")

    email_archive = pd.read_excel(
        email_archive_path, sheet_name="mail_archive_old_format"
    )
    year_mask = (email_archive["year"] >= 2001) & (email_archive["year"] <= 2013)
    # year_mask = (email_archive["year"] == 2011) & (email_archive["month"] == 1)
    masked_email_archive = email_archive[year_mask].reset_index(drop=True)

    base_url = "https://www.unicode.org/mail-arch/unicode-ml/"
    extractor = EmailExtractor(base_url)
    expanded_df = extractor.process_archive(masked_email_archive, max_workers=8)

    # Clean illegal characters before saving to Excel
    expanded_df = EmailExtractor.clean_dataframe_strings(expanded_df)

    output_path = os.path.join(base_path, "utc_email_old_archive_parsed.xlsx")
    expanded_df.to_excel(output_path, index=False, engine="openpyxl")


if __name__ == "__main__":
    main()
