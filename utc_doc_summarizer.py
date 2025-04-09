import os
import yake
import pandas as pd
from rake_nltk import Rake
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser


# import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')

base_path = os.getcwd()
folder_name = "extracted_texts"
folder_path = os.path.join(base_path, folder_name)

file_name = "utc_register_with_text.xlsx"
file_path = os.path.join(base_path, file_name)

df = pd.read_excel(file_path).sample(5)
# df = df[df["error_message"].isnull()].reset_index(drop=True)  # Remove or comment out filtering

df["rake_keywords"] = None
df["yake_keywords"] = None
df["lsa_summary"] = None

def read_file_content(doc_num, folder_path):
    doc_num = doc_num.replace("/", "_")
    file_path = os.path.join(folder_path, f"{doc_num}.txt")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except:
        return ""

def get_rake_keywords(text, proportion=0.01):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()
    return keywords[:int(len(keywords) * proportion)]

def get_yake_keywords(text, proportion=0.01):
    yake_ext = yake.KeywordExtractor(lan="en", n=3)
    keywords = yake_ext.extract_keywords(text)
    return keywords[:int(len(keywords) * proportion)]

def get_lsa_summary(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=sentences_count)
    return " ".join(str(sentence) for sentence in summary)

for idx, row in df.iterrows():
    if pd.isnull(row["error_message"]):
        text = read_file_content(row["doc_num"], folder_path)
        df.at[idx, "text"] = text
        df.at[idx, "rake_keywords"] = get_rake_keywords(text, 0.01)
        df.at[idx, "yake_keywords"] = get_yake_keywords(text, 0.01)
        df.at[idx, "lsa_summary"] = get_lsa_summary(text, 5)
    else:
        df.at[idx, "text"] = ""
        df.at[idx, "rake_keywords"] = None
        df.at[idx, "yake_keywords"] = None
        df.at[idx, "lsa_summary"] = None

output_file_name = "utc_register_with_text_and_summary.xlsx"
df.to_excel(os.path.join(base_path, output_file_name), index=False)