import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action = 'ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from gensim.corpora.dictionary import Dictionary
from gensim.models import Phrases

import koreanToken

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Content:
    def __init__(self, df):
        self.df = df
        self.cosine_sim, self.indices = self.base()
        self.news = input("보고싶은 뉴스를 입력하세요! ")  # 이 입력은 데이터베이스로부터 현재 읽은 뉴스 제목을 받아옴.
        self.rec()

    def base(self):
        # TF-IDF Vectorizer Object 정의. stopword 제거(쓸모 없지만 자주 사용되는 단어는 제거)
        tfidf = TfidfVectorizer()
        #  NaN을 empty string으로 변환
        self.df['content'] = self.df['content'].fillna('')
        #  content 항목에 대한 TF-IDF matrix 생성
        tfidf_matrix = tfidf.fit_transform(self.df['content'])

        # (TFIDF vector 이용할 것이므로) linear_kernel을 대신 이용하여 dot product 계산
        # cosine similarity matrix와 같은 결과
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # index와 news title에 대해 reverse mapping 후 중복된 title 제거
        indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()

        return cosine_sim, indices

    def content_recommender(self):
        # title에 해당하는 news index
        idx = self.indices[self.news]

        # 해당 news를 기준으로  pairwsie similarity score 계산 후  convert into a list of tuples
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # cosine similarity scores 기준으로 정렬
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 10 most similar news의 score. 단, first news 제외(왜냐하면 similarity score  with itself)
        sim_scores = sim_scores[1:5]  # cosine 유사도가 높은 1~10까지의 기사

        # news indices 생성
        news_indices = [i[0] for i in sim_scores]

        # top 10 most similar news
        return self.df['title'].iloc[news_indices], news_indices

    # 추천항목 도출
    def rec(self):
        title, index = self.content_recommender()
        i = 1
        for a in index:
            print(str(i) + ') ' + str(title[a] + '\n'))  # 글자 삽입
            i += 1


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
        self.num_clusters = 2  # 2개의 그룹으로 분류
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

