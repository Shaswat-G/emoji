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
        """Clean text to avoid Unicode encoding issues"""
        if not text or not isinstance(text, str):
            return text
            
        try:
            # Normalize Unicode (NFC form tends to work best)
            text = unicodedata.normalize('NFC', text)
            
            # Remove or replace problematic characters
            # Replace surrogate pairs with replacement character
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            return text
        except Exception:
            # If all else fails, return a safe string
            return "[Encoding Error]"

    def extract_email_from_soup(self, soup):
        # Extract metadata from HTML head
        title = soup.title.text.strip() if soup.title else ""

        # Extract metadata from the headers div
        meta_author = soup.find("meta", {"name": "Author"})
        author = meta_author["content"] if meta_author else ""

        meta_subject = soup.find("meta", {"name": "Subject"})
        subject = meta_subject["content"] if meta_subject else ""

        meta_date = soup.find("meta", {"name": "Date"})
        date_str = meta_date["content"] if meta_date else ""

        # Extract headers from HTML comments in the .head div
        head_div = soup.find("div", {"class": "head"})
        html_comments = []
        if head_div:
            for element in head_div.children:
                if isinstance(element, type(soup.comment)):
                    html_comments.append(str(element))
                elif hasattr(element, "string") and isinstance(
                    element.string, type(soup.comment)
                ):
                    html_comments.append(str(element.string))
        # Parse headers from comments
        headers = {}
        for comment in html_comments:
            matches = re.findall(r'([a-zA-Z0-9_-]+)="([^"]*)"', comment)
            for k, v in matches:
                headers[k.lower()] = v

        # Extract email headers and body
        mail_div = soup.find("div", {"class": "mail"})
        if not mail_div:
            return None

        # Get more detailed headers from the mail div
        from_address = ""
        from_name = ""
        to_address = ""
        cc_address = ""
        bcc_address = ""
        message_id = headers.get("id", "")
        in_reply_to = headers.get("inreplyto", "")
        references = headers.get("references", "")

        # From
        from_span = mail_div.find("span", {"id": "from"})
        if from_span:
            email_link = from_span.find("a")
            if email_link:
                mailto = email_link.get("href", "")
                match = re.search(r"mailto:([^?]+)", mailto)
                if match:
                    from_address = match.group(1).replace("_at_", "@")
                else:
                    from_address = email_link.text.strip().replace("_at_", "@")
            from_text = from_span.get_text(" ", strip=True)
            name_match = re.match(r"From\s*:\s*(.*?)\s*<", from_text)
            if name_match:
                from_name = name_match.group(1).strip()

        # Try to extract To, Cc, Bcc from <address class="headers"> if present
        address_block = mail_div.find("address", {"class": "headers"})
        if address_block:
            # Look for spans or lines containing To, Cc, Bcc
            for line in address_block.stripped_strings:
                # To
                to_match = re.match(r"To\s*: (.+)", line)
                if to_match:
                    to_address = to_match.group(1)
                # Cc
                cc_match = re.match(r"Cc\s*: (.+)", line)
                if cc_match:
                    cc_address = cc_match.group(1)
                # Bcc
                bcc_match = re.match(r"Bcc\s*: (.+)", line)
                if bcc_match:
                    bcc_address = bcc_match.group(1)
        # Fallback: try to extract To, Cc, Bcc from HTML comments
        if not to_address:
            to_address = headers.get("to", "")
        if not cc_address:
            cc_address = headers.get("cc", "")
        if not bcc_address:
            bcc_address = headers.get("bcc", "")

        # Date
        date_span = mail_div.find("span", {"id": "date"})
        date_value = (
            date_span.find(string=True, recursive=False).strip() if date_span else ""
        )
        date_value = date_value.lstrip(":").strip()
        date = date_value or headers.get("sent", "") or headers.get("date", "") or ""

        # Received
        received = ""
        received_span = mail_div.find("span", {"id": "received"})
        if received_span:
            received_text = received_span.get_text(strip=True)
            match = re.search(r"Received on .+ - (.+)", received_text)
            if match:
                received = match.group(1)
        if not received:
            received = headers.get("received", "")

        # Message-ID, In-Reply-To, References from HTML comments
        if not message_id:
            message_id = headers.get("mid", "") or headers.get("message-id", "")
        if not in_reply_to:
            in_reply_to = headers.get("in-reply-to", "") or headers.get("inreplyto", "")
        if not references:
            references = headers.get("references", "")

        # Try to extract in-reply-to and references from navigation links in <div class="foot">
        foot_div = soup.find("div", {"class": "foot"})
        if foot_div:
            # In-Reply-To
            in_reply_to_link = None
            for dfn in foot_div.find_all("dfn"):
                if dfn.get_text(strip=True).lower() == "in reply to":
                    next_a = dfn.find_next("a")
                    if next_a and next_a.has_attr("href"):
                        in_reply_to_link = next_a["href"]
            if in_reply_to_link and not in_reply_to:
                in_reply_to = in_reply_to_link
            # References (all previous messages in thread)
            references_links = []
            for dfn in foot_div.find_all("dfn"):
                if dfn.get_text(strip=True).lower() == "previous message":
                    prev_a = dfn.find_next("a")
                    if prev_a and prev_a.has_attr("href"):
                        references_links.append(prev_a["href"])
            if references_links and not references:
                references = ", ".join(references_links)

        # Extract the message body
        message_body = ""
        body_p = mail_div.find("p")
        if body_p:
            for element in body_p.contents:
                if getattr(element, "name", None) == "br":
                    message_body += "\n"
                else:
                    message_body += str(element).strip()
            message_body = re.sub(r"<!--nospam-->", "", message_body)
            message_body = re.sub(r"<[^>]+>", "", message_body)

        email_message = {
            "subject": (subject or title).strip(),
            "from": (from_address or author.replace("_at_", "@")).strip(),
            "from_name": from_name.strip(),
            "to": to_address.strip(),
            "cc": cc_address.strip(),
            "bcc": bcc_address.strip(),
            "date": date.strip(),
            "received": received.strip(),
            "message_id": message_id.strip(),
            "in_reply_to": in_reply_to.strip(),
            "references": references.strip(),
            "body": message_body.strip(),
            # Optionally add more headers if needed
        }
        return email_message

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
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            try:
                email_message = self.extract_email_from_soup(soup)
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
    masked_email_archive = email_archive[year_mask].reset_index(drop=True)

    base_url = "https://www.unicode.org/mail-arch/unicode-ml/"
    extractor = EmailExtractor(base_url)
    expanded_df = extractor.process_archive(masked_email_archive, max_workers=8)

    output_path = os.path.join(base_path, "utc_email_old_archive_parsed.xlsx")
    expanded_df.to_excel(output_path, index=False, engine="openpyxl")


if __name__ == "__main__":
    main()
