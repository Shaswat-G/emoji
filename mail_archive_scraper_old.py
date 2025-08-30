# -----------------------------------------------------------------------------
# Script: mail_archive_scraper_old.py
# Purpose: Scrape and parse Unicode mailing list archives (2011-2021) to extract
#          email metadata, handling both old (pre-2011-06) and new (post-2011-06)
#          HTML formats for heterogeneous archive support.
# Behavior: Fetches archive pages via requests, parses with BeautifulSoup,
#           extracts author, subject, thread URL, and date/time, then saves
#           aggregated data to CSV. Logs progress and errors.
# Inputs:  None (hardcoded URL pattern for unicode.org/mail-arch/unicode-ml/).
# Outputs: mail_archive.csv with columns: year, month, author, thread, subject, time.
# Notes:   Designed for batch scraping; handles request timeouts and parsing errors.
#          Covers months 01-12 for years 2011-2021.
# Requires: requests, beautifulsoup4, pandas, lxml (for BeautifulSoup parser).
# Updated: 2025-08-30
# -----------------------------------------------------------------------------


import logging

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_mail_data_old_format(soup):
    """Extract mail data from archives before 2011-06 (old format)"""
    data = []
    ulist = soup.find("ul")
    if not ulist:
        return data

    items = ulist.find_all("li")

    for item in items:
        # Extract subject and thread from the first anchor with a <strong> child
        subj_anchor = item.find("a", href=True, text=True)
        strong_tag = item.find("strong")
        if subj_anchor and strong_tag:
            subject = strong_tag.get_text(strip=True)
            thread = subj_anchor.get("href")
        else:
            continue

        em_tags = item.find_all("em")
        # Expect em[0] to be author and em[1] to be time
        if len(em_tags) >= 2:
            author = em_tags[0].get_text(strip=True)
            time_val = em_tags[1].get_text(strip=True).strip("()")
        else:
            author = ""
            time_val = ""

        data.append(
            {"author": author, "thread": thread, "subject": subject, "time": time_val}
        )
    return data


def extract_mail_data_new_format(soup):
    """Extract mail data from archives from 2011-06 onwards (new format)"""
    data = []
    messages_div = soup.find("div", class_="messages-list")
    # print(messages_div)
    if not messages_div:
        return data

    date_items = messages_div.find_all("ul", recursive=False)[0].find_all(
        "li", recursive=False
    )
    # print(date_items)

    for date_item in date_items:
        # Extract the date from the dfn element
        dfn = date_item.find("dfn")
        if not dfn:
            continue

        date_str = dfn.get_text(strip=True)

        # Process all messages for this date
        message_list = date_item.find("ul")
        if not message_list:
            continue

        message_items = message_list.find_all("li")
        for message in message_items:
            # Extract subject and thread link from first anchor
            subject_link = message.find("a", href=True)
            if not subject_link:
                continue

            subject = subject_link.get_text(strip=True)
            thread = subject_link.get("href")

            # Extract author from em tag
            author_tag = message.find("em")
            author = author_tag.get_text(strip=True) if author_tag else ""

            data.append(
                {
                    "author": author,
                    "thread": thread,
                    "subject": subject,
                    "time": date_str,
                }
            )
    return data


def extract_mail_data(url, year, month):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        if year > 2011 or (year == 2011 and month >= 6):
            logging.info(f"Processing {year}-{month:02d} using new format")
            return extract_mail_data_new_format(soup)
        else:
            logging.info(f"Processing {year}-{month:02d} using old format")
            return extract_mail_data_old_format(soup)

    except Exception as e:
        logging.error(f"Error extracting data from {url}: {str(e)}")
        return []


if __name__ == "__main__":
    # year = 2011
    # month = 6
    # month_str = f"m{month:02d}"
    # archive_suffix = f"y{year}-{month_str}"
    # url = f"https://www.unicode.org/mail-arch/unicode-ml/{archive_suffix}/"
    # mail_data = extract_mail_data(url, year, month)
    # print(mail_data)

    all_data = []
    # Loop over years 2011 to 2021 and months 01 to 12
    for year in range(2011, 2022):
        for month in range(1, 13):
            try:
                month_str = f"m{month:02d}"
                archive_suffix = f"y{year}-{month_str}"
                url = f"https://www.unicode.org/mail-arch/unicode-ml/{archive_suffix}/"

                logging.info(f"Processing archive: {archive_suffix}")

                # Extract mail data using the appropriate format
                mail_data = extract_mail_data(url, year, month)
                logging.info(f"Found {len(mail_data)} emails in {archive_suffix}")

                # Add year and month to each record
                for record in mail_data:
                    record["year"] = year
                    record["month"] = month

                all_data.extend(mail_data)

            except RequestException as e:
                logging.error(f"Error requesting {year}-{month}: {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error processing {year}-{month}: {str(e)}")
                continue

    if all_data:
        df = pd.DataFrame(all_data)

        # Ensure 'year' and 'month' are extreme left columns
        cols = ["year", "month"]
        other_cols = [col for col in df.columns if col not in cols]
        df = df[cols + other_cols]

        df.to_csv("mail_archive.csv", index=False)
        logging.info(f"CSV file saved as mail_archive.csv with {len(df)} records")
    else:
        logging.warning("No data was collected, CSV file not created")
