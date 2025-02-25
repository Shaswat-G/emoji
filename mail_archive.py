import requests
from bs4 import BeautifulSoup
import pandas as pd

def extract_mail_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    ulist = soup.find('ul')
    items = ulist.find_all('li')
    
    data = []
    for item in items:
        # Extract subject and thread from the first anchor with a <strong> child
        subj_anchor = item.find('a', href=True, text=True)
        strong_tag = item.find('strong')
        if subj_anchor and strong_tag:
            subject = strong_tag.get_text(strip=True)
            thread = subj_anchor.get('href')
        else:
            continue
        
        em_tags = item.find_all('em')
        # Expect em[0] to be author and em[1] to be time
        if len(em_tags) >= 2:
            author = em_tags[0].get_text(strip=True)
            time_val = em_tags[1].get_text(strip=True).strip("()")
        else:
            author = ""
            time_val = ""
        
        data.append({
            'author': author,
            'thread': thread,
            'subject': subject,
            'time': time_val
        })
    return data

if __name__ == '__main__':
    all_data = []
    # Loop over years 2011 to 2021 and months 01 to 12
    for year in range(2011, 2022):
        for month in range(1, 13):
            month_str = f"m{month:02d}"
            archive_suffix = f"y{year}-{month_str}"
            url = f"https://www.unicode.org/mail-arch/unicode-ml/{archive_suffix}/"
            mail_data = extract_mail_data(url)
            for record in mail_data:
                record['year'] = year
                record['month'] = month
            all_data.extend(mail_data)
    
    df = pd.DataFrame(all_data)
    # Ensure 'year' and 'month' are extreme left columns
    cols = df.columns.tolist()
    for col in ['year', 'month']:
        cols.remove(col)
    df = df[['year', 'month'] + cols]
    
    df.to_csv("mail_archive.csv", index=False)
    print("CSV file saved as mail_archive.csv")