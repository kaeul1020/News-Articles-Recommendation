from recommend_engine import Process, Cluster, Sentiment
import pandas as pd

# 이 파일을 실행하세요

# 사용자가 본 뉴스
data = pd.read_excel('News-Articles-Recommendation/ko_model/news.xlsx')
data = data[data['valid']==1]
lines = data['text'].tolist()

pc = Process()
Str = pc.go(lines)

Clu = Cluster(Str)
clusters = Clu.division()

df = pd.DataFrame(columns=['cluster', 'texts', 'ptexts', 'corpus', 'docs'])
df['cluster'] = clusters
df['texts'] = data['text']
df['ptexts'] = Str


Sent = Sentiment(df['ptexts'])
corpus, docs, dictionary = Sent.Corpus()

df['corpus'] = corpus
df['docs'] = docs

UniqueNames = df['cluster'].unique()


#create a data frame dictionary to store data frames
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.wrappers import LdaMallet
import numpy as np
from gensim.corpora.dictionary import Dictionary

DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}
docscluster={elem : pd.DataFrame for elem in UniqueNames}
corpuscluster={elem : pd.DataFrame for elem in UniqueNames}
dictionarycluster={elem : pd.DataFrame for elem in UniqueNames}


for key in DataFrameDict.keys():
    DataFrameDict[key] = df[:][df.cluster == key]
    docscluster[key]=df['docs'][df.cluster == key]
    corpuscluster[key]=df['corpus'][df.cluster == key]
    dictionarycluster[key]=Dictionary(docscluster[key])


# polarity(긍정적/부정적 감정 : -1~1 사이의 값), subjectivity(객관성/주관성 : 0~1 사이의 값)
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import nltk

polaritycluster={elem : pd.DataFrame for elem in UniqueNames}
subjectivitycluster={elem : pd.DataFrame for elem in UniqueNames}
for i in DataFrameDict.keys():
    polaritycluster[i]=TextBlob(' '.join(DataFrameDict[i]['texts'].astype('str'))).sentiment.polarity
    subjectivitycluster[i]=TextBlob(' '.join(DataFrameDict[i]['texts'].astype('str'))).sentiment.subjectivity


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print('-'*100)
    print('User가 본 뉴스에 대한 정보를 담고 있는 df 데이터프레임')
    print('-'*100)
    print(df)




# 이 부분이 토픽 모델링 부분(LdaMallet 77번 줄 에서 오류가 발생.....)
# 토픽을 네이버 뉴스에 있는 정치/사회/생활 등으로 직접 지정해주면 해결할 수 있을 것 같기도 함.
# import os
# os.environ['MALLET_HOME'] = 'C:\\new_mallet\\mallet-2.0.8'

# mallet_path = 'C:\\new_mallet\\mallet-2.0.8\\bin\\mallet'
# ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=3, id2word=dictionary)
# print(ldamallet.show_topics(formatted=False))

# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=docs, dictionary=dictionary, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: ', coherence_ldamallet)

# 최적의 토픽 수를 찾기 위해 여러 토픽 수로 일관성을 계산하고 비교
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values


# model_list={elem : pd.DataFrame for elem in UniqueNames }
# coherence_values={ elem : pd.DataFrame for elem in UniqueNames}
# for i in np.arange(Clu.num_clusters):
#     model_list[i], coherence_values[i] = compute_coherence_values(dictionary=dictionarycluster[i], corpus=corpuscluster[i], texts=docscluster[i], start=2, limit=6, step=1)


# # Print the coherence scores
# import math
# best_value=np.array([])
# optimal_model={elem : pd.DataFrame for elem in UniqueNames }
# for i in np.arange(Clu.num_clusters):
#     coherence_values[i]=[value for value in coherence_values[i] if not math.isnan(value)]
#     best_value=np.append(best_value, np.amax(coherence_values[i]))
#     t=int(np.argmax(coherence_values[i]))
#     optimal_model[i]=model_list[i][t]
#     print("Cluster=",i,"has optimal number of topics as", t+2)





#-----------------------------------------------------------------
# 새로운 뉴스, 추천할 뉴스 골라내기
data2 = pd.read_excel('News-Articles-Recommendation/ko_model/news.xlsx')
data2 = data2[data2['valid']==0]
lines2 = data2['text'].tolist()

pc2 = Process()
Str2 = pc2.go(lines2)

df_text = pd.DataFrame(columns=['text'])
df_text['text'] = Str2

Sent2 = Sentiment(df_text['text'])
corpus2, docs2, dictionary2 = Sent2.Corpus()

pol=[TextBlob(' '.join(df_text.iloc[i,0])).sentiment.polarity for i in range(df_text.shape[0])]
sub=[TextBlob(' '.join(df_text.iloc[i,0])).sentiment.subjectivity for i in range(df_text.shape[0])]
df_text['pol']=pol
df_text['sub']=sub

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print('-'*100)
    print('-'*100)
    print('새로운 뉴스에 대한 정보(이 중에 계산을 통해 추천을 해줄 것)')
    print('-'*100)
    print(df_text)
    print('-'*100)




# 토픽 모델링 부분을 해결해야 이것도 실행할 수 있음
# Function to analyze the developed topic models on unseen corpus of texts (in our case News articles)
# def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=docs):
#     # Init output
#     sent_topics_df = pd.DataFrame()

#     # Get main topic in each document
#     for i, row in enumerate(ldamodel[corpus]):
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)
#         # Get the Dominant topic, Perc Contribution and Keywords for each document
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:  # => dominant topic
#                 wp = ldamodel.show_topic(topic_num)
#                 topic_keywords = ", ".join([word for word, prop in wp])
#                 sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
#             else:
#                 break
#     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
#     # Add original text to the end of the output
#     contents = pd.Series(texts)
#     sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
#     return(sent_topics_df)


# # The percent contribution of each topic model considered as metric to assign topic score
# df_topic_sents_keywords={elem : pd.DataFrame for elem in UniqueNames}
# topicss=pd.DataFrame()
# for i in range(len(UniqueNames)):
#     mod=gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model[i])
#     temp={'cluster':i,
#          'model':mod}
#     topicss=topicss.append(temp, ignore_index=True)
#     df_topic_sents_keywords[i] = format_topics_sentences(ldamodel=mod, corpus=corpus, texts=docs)

#     # Centre the percent contribution by subtracting the mean. This standardises topic score
#     df_topic_sents_keywords[i]['Diff']= df_topic_sents_keywords[i]['Perc_Contribution']-np.mean(df_topic_sents_keywords[i]['Perc_Contribution'])

# sentiment={elem: pd.DataFrame for elem in UniqueNames}
# subjectivit={elem: pd.DataFrame for elem in UniqueNames}
# w1=0.8 # Topic modelling weight
# w2=0.2 # Sentiment score weight
# for i in range(len(UniqueNames)):
#     sentiment[i]=cosine_similarity(np.array(df_text.iloc[:, 2]).reshape(-1, 1),np.array([polaritycluster[i]]).reshape(-1, 1))
#     subjectivit[i]=cosine_similarity(np.array(df_text.iloc[:, 2]).reshape(-1, 1),np.array([subjectivitycluster[i]]).reshape(-1, 1))
#     df_topic_sents_keywords[i]['Polarity']=sentiment[i]
#     df_topic_sents_keywords[i]['Subjectivity']=subjectivit[i]
#     df_topic_sents_keywords[i]['Metric']=w1*df_topic_sents_keywords[i]['Diff']+w2/2*(df_topic_sents_keywords[i]['Polarity']+df_topic_sents_keywords[i]['Subjectivity'])



# # 추천 메트릭스
# recommend=pd.DataFrame()
# metric_value=pd.DataFrame()
# rec=np.array([])
# for i in range(len(docs)):
#     for j in range(len(UniqueNames)):
#         rec=np.append(rec, df_topic_sents_keywords[j].iloc[i,7])

#     recommend=recommend.append(pd.Series(np.argmax(rec)),ignore_index=True)
#     metric_value=metric_value.append(pd.Series(np.amax(rec)),ignore_index=True)
#     rec=np.array([])

# recommend['metric']=metric_value
# recommend['url']=df_text['Link']
# recommend['article_text']=df_text['text']
# recommend.rename(columns={0:'cluster'},inplace=True)
# recommend.to_csv('%srecommend.csv'%location,index=None)



