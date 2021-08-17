import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action = 'ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases

import koreanToken

class Process:
    def __init__(self):
        self.to_corpus = []
        self.Str = []

    def go(self, lines):

        for line in lines:
            self.Token(line)

        for text in self.to_corpus:
            StrA = " ".join(text)
            self.Str.append(StrA)

        return self.Str

    def Token(self, line):
        page = koreanToken.BasicProcessing()
        token = page.Tokenize(line)

        page2 = koreanToken.PartOfSpeech()
        pos_tagged_corpus = page2.pos_text(token)

        page3 = koreanToken.Stemming()
        stemming_corpus = page3.stemming_text(pos_tagged_corpus)

        page4 = koreanToken.Stopwords()
        removed_stopword_corpus = page4.remove_stopword_text(stemming_corpus)

        self.to_corpus.append(removed_stopword_corpus)



class Cluster:
    def __init__(self, Str):
        self.series_data = pd.Series(Str)
        self.num_clusters = 3  # 3개의 그룹으로 분류
        self.cluster = []

    def division(self):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000, min_df=0.1, use_idf=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.series_data)

        km = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=1)

        km.fit(tfidf_matrix)
        self.clusters = km.labels_.tolist()

        return self.clusters



class Sentiment:
    def __init__(self, text):
        self.docs = list(text.apply(lambda x: x.split()))
        self.bigram = Phrases(self.docs, min_count=10)
        self.trigram = Phrases(self.bigram[self.docs])
    
    def Corpus(self):
        for idx in range(len(self.docs)):
            for token in self.bigram[self.docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    self.docs[idx].append(token)
            for token in self.trigram[self.docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    self.docs[idx].append(token)

        dictionary = Dictionary(self.docs)
        corpus = [dictionary.doc2bow(doc) for doc in self.docs]

        return corpus, self.docs, dictionary

