from pathlib import Path
import os
import pandas as pd

excel_dir = Path("parsed_excels")

df = pd.DataFrame()

for xlsx in excel_dir.glob("*.xlsx"):
    if "llm" in xlsx.stem:
        df = pd.concat([df, pd.read_excel(xlsx)], ignore_index=True)
        
df.to_excel("email_archive_llmsweep.xlsx", index=False)