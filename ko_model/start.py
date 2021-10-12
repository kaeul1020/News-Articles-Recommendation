from recommend_engine import Process, Cluster, Sentiment, Content
import pandas as pd
#from konlpy.tag import Okt

# 이 파일을 실행하세요

turn = 10  # 이것이 사용자가 본 뉴스 개수

# 사용자 클러스터화
data = pd.read_excel('news_user1.xlsx', engine='openpyxl')
data = data[data['valid']==1]
lines = data['text'].tolist()

df = pd.read_csv('../output/Article_IT과학_201701_201804.csv', header=None, names=['publish_year', 'catagory', 'publish', 'title', 'content', 'url'])

index_list = list(range(0, 20, 1))
df = pd.DataFrame(df[:20], index=index_list)  # 500개의 뉴스만 가져옴

docs = list(df['content'])  # 본 뉴스만 docs에 넣어서 작업 수행.

if (0 <= turn) & (turn < 10):  # 본 뉴스의 개수가 10개가 되지 않으면 content만으로 추천이 이루어진다.
    Ct = Content(df)

else:
    pc = Process()
    Str = pc.go(lines)

    Clu = Cluster(Str)
    clusters = Clu.division()

    df = pd.DataFrame(columns=['cluster', 'texts', 'ptexts', 'corpus', 'docs'])
    df['cluster'] = clusters
    df['texts'] = lines
    df['ptexts'] = Str


    Sent = Sentiment(df['ptexts'])
    corpus, docs, dictionary = Sent.Corpus()

    df['corpus'] = corpus
    df['docs'] = docs

    UniqueNames = df['cluster'].unique()


    # 클러스터화 한 뒤 분석
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

    polaritycluster={elem : pd.DataFrame for elem in UniqueNames}
    subjectivitycluster={elem : pd.DataFrame for elem in UniqueNames}
    for i in DataFrameDict.keys():
        polaritycluster[i]=TextBlob(' '.join(DataFrameDict[i]['texts'].astype('str'))).sentiment.polarity
        subjectivitycluster[i]=TextBlob(' '.join(DataFrameDict[i]['texts'].astype('str'))).sentiment.subjectivity


    # 사용자 데이터프레임 출력
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('-'*100)
        print('User가 본 뉴스에 대한 정보를 담고 있는 df 데이터프레임')
        print('-'*100)
        print(df)


    # 토픽 모델링
    import tomotopy as tp 
    
    model = tp.LDAModel(k=5, alpha=0.1, eta=0.01, min_cf=4)
    # 전체 말뭉치에 4회 미만 등장한 단어들은 제거할 겁니다.
    
    for line in df['ptexts']:
        model.add_doc(line.strip().split(' ')) # 공백 기준으로 단어를 나누어 model에 추가합니다.


    # model의 num_words나 num_vocabs 등은 train을 시작해야 확정됩니다.
    # 따라서 이 값을 확인하기 위해서 train(0)을 하여 실제 train은 하지 않고 학습 준비만 시킵니다.
    # num_words, num_vocabs에 관심 없다면 이부분은 생략해도 됩니다.
    model.train(0) 
    # print('Total docs:', len(model.docs))
    # print('Total words:', model.num_words)
    # print('Vocab size:', model.num_vocabs)
    
    # 다음 구문은 train을 총 200회 반복하면서, 매 단계별로 로그 가능도 값을 출력해줍니다.
    # 혹은 단순히 model.train(200)으로 200회 반복도 가능합니다.
    for i in range(200):
        model.train(1)
    
    # 학습된 토픽들을 출력해보도록 합시다.
    for i in range(model.k):
        # 토픽별 상위 단어 10개를 뽑아봅시다.
        res = model.get_topic_words(i, top_n=10)
        print('Topic #{}'.format(i), end='\t')
        print(', '.join(w for w, p in res))

    for i in range(len(model.docs)):
        print('doc #{}'.format(i), end='\t')
        print(model.docs[i].get_topics(top_n=2))


    #-----------------------------------------------------------------
    # 새로운 뉴스, 추천할 뉴스 골라내기
    data2 = pd.read_excel('news_user1.xlsx', engine='openpyxl')
    data = data[data['valid']==0]
    lines2 = data2['text'].tolist()

    pc2 = Process()
    Str2 = pc2.go(lines2)

    df_text = pd.DataFrame(columns=['text'])
    df_text['text'] = Str2

    Sent2 = Sentiment(df_text['text'])
    corpus2, docs2, dictionary2 = Sent.Corpus()

    pol=[TextBlob(' '.join(df_text.iloc[i,0])).sentiment.polarity for i in range(df_text.shape[0])]

    df_text['pol']=pol

    def format_topics_sentences(num, ldamodel):
        Topic_num_ls = []
        Perc_Contribution_ls = []
        Topic_Keywords_ls = []

        for j in range(len(ldamodel.docs)):
            topic = list(ldamodel.docs[j].get_topics(top_n=1)[0])
            if topic[0] == num:
                topic_keywords = []
                res = ldamodel.get_topic_words(num, top_n=10)
                topic_keywords.append(', '.join(w for w, p in res))

                Topic_num_ls.append(int(topic[0]))
                Perc_Contribution_ls.append(round(topic[1],4))
                Topic_Keywords_ls.append(topic_keywords)
            
        sent_topics_df = pd.DataFrame({'Topic_num': Topic_num_ls, 
                                        'Perc_Contribution': Perc_Contribution_ls, 
                                        'Topic_Keywords': Topic_Keywords_ls})
        return sent_topics_df


    from sklearn.metrics.pairwise import cosine_similarity

    # The percent contribution of each topic model considered as metric to assign topic score
    df_topic_sents_keywords={elem : pd.DataFrame for elem in UniqueNames}

    for i in range(len(UniqueNames)):
        df_topic_sents_keywords[i] = format_topics_sentences(i, model)
        df_topic_sents_keywords[i]['Diff']= df_topic_sents_keywords[i]['Perc_Contribution']-np.mean(df_topic_sents_keywords[i]['Perc_Contribution'])


    sentiment={elem: pd.DataFrame for elem in UniqueNames}

    w1=0.8 # Topic modelling weight
    w2=0.2 # Sentiment score weight
    for i in range(len(UniqueNames)):
        sentiment[i] = cosine_similarity(np.array(df_text.iloc[:, 1]).reshape(-1, 1), np.array([polaritycluster[i]]).reshape(-1, 1))
        
        sentiment[i] = np.array(sentiment[i]).flatten().tolist()
        
        a = []
        b = []
        for j in range(len(model.docs)):
            topic = list(model.docs[j].get_topics(top_n=1)[0])
            if topic[0] == i:
                a.append(sentiment[i][j])

        df_topic_sents_keywords[i]['Polarity'] = a

        df_topic_sents_keywords[i]['Metric']=w1*df_topic_sents_keywords[i]['Diff']+w2/2*(df_topic_sents_keywords[i]['Polarity'])


    print(df_topic_sents_keywords[0])
    print(df_topic_sents_keywords[1])


    # 추천 메트릭스
    recommend=pd.DataFrame()
    recommender=pd.DataFrame()
    metric_value=pd.DataFrame()
    rec=np.array([])


    for j in range(len(model.docs)):
        count = 0
        topic = list(model.docs[j].get_topics(top_n=1)[0])
        for i in range(len(UniqueNames)):  
            if topic[0] == i:
                rec=np.append(rec, df_topic_sents_keywords[i].iloc[count,5])
                count += 1

        recommender=recommender.append(pd.Series(np.argmax(rec)),ignore_index=True)
        metric_value=metric_value.append(pd.Series(np.amax(rec)),ignore_index=True)
        rec=np.array([])

    recommend['cluster']=recommender
    recommend['metric']=metric_value
    recommend['article_text']=df_text['text']
    recommend.to_csv('recommend.csv',index=None)



