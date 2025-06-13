import pandas as pd
import os
import ast

base_path = os.getcwd()


mapper_name = "emoji_to_proposal_map.csv"
mapper_path = os.path.join(base_path, mapper_name)
mapper = pd.read_csv(mapper_path)
mapper["proposal_doc_num"] = mapper["proposal_doc_num"].apply(lambda x: x.split(",") if pd.notnull(x) else [])


proposal_name = "emoji_proposal_table.csv"
proposal_path = os.path.join(base_path, proposal_name)
proposals = pd.read_csv(proposal_path)

utc_doc_name = "utc_register_all_classified.xlsx"
utc_doc_path = os.path.join(base_path, utc_doc_name)
utc_docs = pd.read_excel(utc_doc_path)

# Convert the 'doc_type' column from string representation of dict to actual Python dict
utc_docs["doc_type"] = utc_docs["doc_type"].apply(
    lambda x: dict(ast.literal_eval(x)) if pd.notnull(x) else {} if pd.notnull(x) else {}
)

def is_proposal(doc_type):
    return 1 if "Proposals" in doc_type else 0

utc_docs["proposals"] = utc_docs["doc_type"].apply(is_proposal)
print(utc_docs["proposals"].value_counts())

# total_proposals = set(proposals["doc_num"].tolist())
# accepted_proposals = set(doc_num for sublist in mapper["proposal_doc_num"] for doc_num in sublist if doc_num)


# print(total_proposals - accepted_proposals)
