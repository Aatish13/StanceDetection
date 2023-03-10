import contractions
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')

def convert_to_lower(text):
    return text.lower()

def clean_ascii(text):
    text = re.sub("[^a-z0-9]"," ", text)
    return text

def remove_mentions(text):
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub("#[A-Za-z0-9_]+","", text)
    return text

def remove_links(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www.\S+", "", text)
    return text

def remove_punctuation(text):
    text = re.sub('[()!?]', ' ', text)
    text = re.sub('\[.*?\]',' ', text)
    return text
  
def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_sentence = [w for w in text.split() if not w in stopwords]
    return " ".join(filtered_sentence)

def open_contractions(text):
  return contractions.fix(text)