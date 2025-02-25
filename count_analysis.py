import pandas as pd

df_reject = pd.read_csv('reject_count.csv')
df_reject["Year"] = df_reject["Date"].apply(lambda x: int(x[0:4]))
df_reject["Document_code"] = df_reject["Document"].apply(lambda x: x[0:9])

df_accept = pd.read_csv('proposal_count.csv')
df_accept["Year"] = df_accept["Col1"].apply(lambda x: int("20"+x[3:5]))

year_start = 2011
year_end = 2021
df_reject = df_reject[(df_reject["Year"] >= year_start) & (df_reject["Year"] <= year_end)]
df_accept = df_accept[(df_accept["Year"] >= year_start) & (df_accept["Year"] <= year_end)]

print(df_reject.head())
print(df_accept.head())

accepted_proposals = set(df_accept['Col1'])
print(df_accept.shape, len(accepted_proposals))

rejected_proposals = set(df_reject['Document_code'])
print(df_reject.shape, len(rejected_proposals))

total_proposals = accepted_proposals.union(rejected_proposals)
print("Total proposals:", len(total_proposals))