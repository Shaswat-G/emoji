import os
import pandas as pd
import json


base_path = os.getcwd()
emoji_proposal_path = os.path.join(base_path, "emoji_proposal_table.json")
emoji_proposal_df = pd.read_json(emoji_proposal_path, orient="records")
emoji_proposal_df.drop(columns=["Emoji Image"], inplace=True)

emoji_proposal_df.to_csv(
    os.path.join(base_path, "emoji_proposal_table.csv"),
    index=False,
    encoding="utf-8",
)
