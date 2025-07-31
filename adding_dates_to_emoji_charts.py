# -----------------------------------------------------------------------------
# Script: adding_dates_to_emoji_charts.py
# Summary: Enriches emoji chart data by merging with Unicode technical release
#          date mappings to add precise version and date information for
#          temporal analysis of emoji standardization patterns.
# Inputs:  emoji_chart_extracted.xlsx, year_to_technical_release_date_map.xlsx
# Outputs: emoji_chart_with_dates.xlsx
# Context: Data preparation step in emoji proposal research pipeline, enabling
#          chronological analysis of Unicode emoji adoption and release timing
#          for academic study of standardization processes.
# -----------------------------------------------------------------------------

import os
import pandas as pd


em_chart = os.path.join(os.path.dirname(__file__), "emoji_chart_extracted.xlsx")
date_map = os.path.join(os.path.dirname(__file__), "year_to_technical_release_date_map.xlsx")

em_df = pd.read_excel(em_chart)
date_map_df = pd.read_excel(date_map)


# Assume date_map_df has columns: Year, Date, Version (case-insensitive)
# Standardize column names for merge
date_map_df.columns = [c.strip().capitalize() for c in date_map_df.columns]

# Ensure both Year columns are string type for merge
em_df["Year"] = em_df["Date"].astype(str)
date_map_df["Year"] = date_map_df["Year"].astype(str)

merged = em_df.merge(date_map_df, left_on="Year", right_on="Year", how="left")

# Reorder columns: Version, Year, Date (exact), ...rest...
merged = merged.rename(columns={"Date_y": "Date_exact", "Version": "Version"})
cols = ["Version", "Year", "Date_exact"] + [
    c
    for c in merged.columns
    if c not in ["Version", "Year", "Date_exact", "Date_x", "Date_y"]
]
merged = merged[cols]

merged = merged.rename(columns={"Date_exact": "Date"})

merged.to_excel("emoji_chart_with_dates.xlsx", index=False)
