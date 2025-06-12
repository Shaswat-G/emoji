import pandas as pd
import os

base_path = os.getcwd()
mapper_name = "emoji_to_proposal_map.csv"
mapper_path = os.path.join(base_path, mapper_name)

proposal_name = "emoji_proposal_table.csv"
proposal_path = os.path.join(base_path, proposal_name)

mapper = pd.read_csv(mapper_path)
proposals = pd.read_csv(proposal_path)
total_proposals = set(proposals["doc_num"].tolist())

mapper["proposal_doc_num"] = mapper["proposal_doc_num"].apply(lambda x: x.split(",") if pd.notnull(x) else [])
accepted_proposals = set(doc_num for sublist in mapper["proposal_doc_num"] for doc_num in sublist if doc_num)

print(accepted_proposals-total_proposals)
