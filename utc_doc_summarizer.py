import os
import pandas as pd
from rake_nltk import Rake
import yake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

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
yake_ext = yake.KeywordExtractor(lan="en", n=3)


rake_summary_proportion = 0.01
yake_summary_proportion = 0.01
lsa_summary_num_sentences = 5

for index, row in df.iterrows():
    doc_num = row["doc_num"].replace('/', '_')
    file_name = f"{doc_num}.txt"
    file_path = os.path.join(folder_path, file_name) 
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        try:
            rake.extract_keywords_from_text(text)
            rake_keywords = rake.get_ranked_phrases()
            rake_volume = len(rake_keywords)
            num_phrases = int(rake_volume * rake_summary_proportion)
            rake_keywords = rake_keywords[:num_phrases]
            print(f"Keywords for {file_name} : {rake_keywords}")
        except Exception as e:
            print(f"Error extracting rake keywords from {file_name}: {e}")
            
        try:
            yake_keywords = yake_ext.extract_keywords(text)
            yake_volume = len(yake_keywords)
            num_phrases = int(yake_volume * yake_summary_proportion)
            yake_keywords = yake_keywords[:num_phrases]
            print(f"Keywords for {file_name} : {yake_keywords}")
        except Exception as e:
            print(f"Error extracting yake keywords from {file_name}: {e}")
            
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=lsa_summary_num_sentences)
            summary_text = " ".join(str(sentence) for sentence in summary)
            print(f"Summary for {file_name} : {summary_text}")
        except Exception as e:
            print(f"Error extracting sumy summary from {file_name}: {e}")
            
    except Exception as e:
        print(f"Could not open file : {file_name} because of Error : {e}.")