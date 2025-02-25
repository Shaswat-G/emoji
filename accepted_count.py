import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

print("Current Working Directory " , os.getcwd())

url = "https://www.unicode.org/emoji/charts/emoji-proposals.html"

response = requests.get(url)
response.raise_for_status() 
soup = BeautifulSoup(response.content, 'html.parser')

# print(soup.prettify())

tables = soup.find_all('table')
if len(tables) >= 2:
    print("Table 1 rows:", len(tables[0].find_all('tr')))
    print("Table 2 rows:", len(tables[1].find_all('tr')))
else:
    print("Expected at least 2 tables in the document.")

# Process the first table: extract first 4 tds without header
table1 = tables[0]
rows = table1.find_all('tr')
data = []
for row in rows:
    cells = row.find_all('td')
    cells_text = [cell.get_text(strip=True) for cell in cells][:4]
    if len(cells_text) == 4:
        data.append(cells_text)


df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3", "Col4"])
df.to_csv('proposal_count.csv', index=False)
print("Data exported to proposal_count.csv")