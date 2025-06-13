# -----------------------------------------------------------------------------
# Script: utc_new_email_concatenator.py
# Summary: Concatenates all LLM-processed email Excel files in parsed_excels/
#          into a single master archive for downstream analysis.
# Inputs:  parsed_excels/*llm*.xlsx (LLM-processed email files)
# Outputs: email_archive_llmsweep.xlsx (concatenated master archive)
# Context: Part of a research pipeline analyzing UTC's emoji proposal and
#          decision-making processes using public data.
# -----------------------------------------------------------------------------

from pathlib import Path
import os
import pandas as pd

excel_dir = Path("parsed_excels")

df = pd.DataFrame()

for xlsx in excel_dir.glob("*.xlsx"):
    if "llm" in xlsx.stem:
        df = pd.concat([df, pd.read_excel(xlsx)], ignore_index=True)

df.to_excel("email_archive_llmsweep.xlsx", index=False)
