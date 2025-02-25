import pandas as pd

df_reject = pd.read_csv('reject_count.csv')
df_accept = pd.read_csv('proposal_count.csv')

accepted_proposals = set(df_accept['Col1'])
print(df_accept.shape, len(accepted_proposals))

rejected_proposals = set(df_reject['Document'])
print(df_reject.shape, len(rejected_proposals))

total_proposals = accepted_proposals.union(rejected_proposals)
print("Total proposals:", len(total_proposals))