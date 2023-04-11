import random
from random import shuffle
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

def apply_synonym(row):
    words=word_tokenize(row)
    n=len(words)
    res=synonym_replacement(words,n//10)
    return ' '.join(res)

def augment_data(df, g,n):
    temp=df[df.stance==g]
    temp1=temp
    for i in range(n):
        temp1.Tweet=temp.Tweet.apply(apply_synonym)
        df=df.append(temp1,ignore_index=True)
    return df

def apply_data_augmentation(df):
    augment_data(df, "neutral",3)
    augment_data(df, "denier",6)

