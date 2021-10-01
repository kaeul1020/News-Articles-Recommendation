# content-based filtering
# 기사 내용을 토대로 하여 유사한 기사 추천(사용자에 대한 정보가 얼마 없을 때는 이 방식으로 뉴스를 누르면 추천이 뜨도록 한다.)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('../output/Article_economy_201701_201804.csv', header=None, names=['publish_year', 'catagory', 'publish', 'title', 'content', 'url'])

index_list = list(range(0, 500, 1))
df = pd.DataFrame(df[:500], index=index_list)  # 500개의 뉴스만 가져옴

docs = list(df['content'])


# TF-IDF Vectorizer Object 정의. stopword 제거(쓸모 없지만 자주 사용되는 단어는 제거)
tfidf = TfidfVectorizer()

#  NaN을 empty string으로 변환
df['content'] = df['content'].fillna('')

#  content 항목에 대한 TF-IDF matrix 생성
tfidf_matrix = tfidf.fit_transform(df['content'])

# (TFIDF vector 이용할 것이므로) linear_kernel을 대신 이용하여 dot product 계산
# cosine similarity matrix와 같은 결과
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# index와 news title에 대해 reverse mapping 후 중복된 title 제거
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# Main 함수
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # title에 해당하는 news index
    idx = indices[title]

    # 해당 news를 기준으로  pairwsie similarity score 계산 후  convert into a list of tuples
    sim_scores = list(enumerate(cosine_sim[idx]))

    # cosine similarity scores 기준으로 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 10 most similar news의 score. 단, first news 제외(왜냐하면 similarity score  with itself)
    sim_scores = sim_scores[1:5]  # cosine 유사도가 높은 1~10까지의 기사

    # news indices 생성
    news_indices = [i[0] for i in sim_scores]

    # top 10 most similar news
    return df['title'].iloc[news_indices], news_indices

# 추천항목 도출
def rec(news):
    title, index = content_recommender(news)
    i = 1
    for a in index:
        print(str(i) + ') ' + str(title[a] + '\n'))  # 글자 삽입
        i += 1


news = input("보고싶은 뉴스를 입력하세요! ")
rec(news)
