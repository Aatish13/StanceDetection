import re  
import pandas as pd  
from time import time  
from collections import defaultdict
import spacy
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import nltk
nltk.download('stopwords')

df = pd.read_csv("../Dataset/Preprocessed_Data.csv")

sent = [row.split() for row in df['Tweet']]


phrases = Phrases(sent, min_count=30, progress_per=10000)
print(phrases)

bigram = Phraser(phrases)

sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1

def get_word_to_vec_model():
    cores = multiprocessing.cpu_count() 

    w2v_model = Word2Vec(min_count=20,
                        window=2,
                        size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)

    w2v_model.build_vocab(sentences, progress_per=10000)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    w2v_model.init_sims(replace=True)
    return w2v_model

def perform_k_means(X):
    sse = []
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

    labels = kmeans.labels_
    word_clusters = {}
    for i, word in enumerate(w2v_model.wv.vocab):
        cluster = labels[i]
        if cluster not in word_clusters:
            word_clusters[cluster] = []
        word_clusters[cluster].append(word)

    return labels

w2v_model = get_word_to_vec_model()
X = w2v_model.wv[w2v_model.wv.vocab]
labels = perform_k_means(X)
silhouette_avg = silhouette_score(X, labels)
db_index = davies_bouldin_score(X, labels)
