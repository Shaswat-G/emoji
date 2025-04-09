import os
import pandas as pd
from rake_nltk import Rake
# import nltk

# nltk.download('stopwords')
# nltk.download('punkt_tab')

base_path = os.getcwd()
folder_name = "extracted_texts"
folder_path = os.path.join(base_path, folder_name)

file_name = "utc_register_with_text.xlsx"
file_path = os.path.join(base_path, file_name)

df = pd.read_excel(file_path).sample(5)
df = df[df["error_message"].isnull()].reset_index(drop=True)

rake = Rake()


rake_summary_proportion = 0.05
for index, row in df.iterrows():
    doc_num = row["doc_num"].replace('/', '_')
    file_name = f"{doc_num}.txt"
    file_path = os.path.join(folder_path, file_name)
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        try:
            rake.extract_keywords_from_text(text)
            keywords_with_scores = rake.get_ranked_phrases()
            volume = len(keywords_with_scores)
            num_phrases = int(volume * rake_summary_proportion)
            keywords_with_scores = keywords_with_scores[:num_phrases]
            print(f"Keywords for {file_name} : {keywords_with_scores}")
        except Exception as e:
            print(f"Error extracting keywords from {file_name}: {e}")
    except Exception as e:
        print(f"Could not open file : {file_name} because of Error : {e}.")