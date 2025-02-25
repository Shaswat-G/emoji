import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

print("Current Working Directory " , os.getcwd())

url = "https://charlottebuff.com/unicode/misc/rejected-emoji-proposals/"

response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')

# Print the soup object to inspect its structure
# print(soup.prettify())

tables = soup.find_all('table')
print(len(tables))
print(len(tables[0].find_all('tr')))

table1 = tables[0]
rows = table1.find_all('tr')
columns = ["Sample", "Name (Meaning)", "Author", "Date", "Document", "Decision"]
prev_values = [""] * len(columns)
data = []

for row in rows:
    cells = row.find_all(['td','th'])
    cells_text = [cell.get_text(strip=True) for cell in cells]
    if len(cells_text) < len(columns):
        cells_text = [cells_text[i] if i < len(cells_text) and cells_text[i] else prev_values[i] for i in range(len(columns))]
    for i in range(len(columns)):
        if cells_text[i]:
            prev_values[i] = cells_text[i]
    data.append(cells_text)

df = pd.DataFrame(data, columns=columns)
df.to_csv('reject_count.csv', index=False)
print("Data exported to reject_count.csv")