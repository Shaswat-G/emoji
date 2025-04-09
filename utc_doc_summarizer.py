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
df = df[df["error_message"].isnull()].reset_index(drop=True)

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

df["text"] = df["doc_num"].apply(lambda x: read_file_content(x, folder_path))
df["rake_keywords"] = df["text"].apply(lambda t: get_rake_keywords(t, 0.01))
df["yake_keywords"] = df["text"].apply(lambda t: get_yake_keywords(t, 0.01))
df["lsa_summary"] = df["text"].apply(lambda t: get_lsa_summary(t, 5))

output_file_name = "utc_register_with_text_and_summary.xlsx"
df.to_excel(os.path.join(base_path, output_file_name), index=False)