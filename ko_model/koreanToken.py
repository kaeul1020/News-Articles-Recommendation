from khaiii import KhaiiiApi 
import kss
import re 


# 토큰화
class BasicProcessing:
    
    def __init__(self):
        self.punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
        self.punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", 
                            "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', 
                            '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    def Tokenize(self, lines):
        
        sentence_tokenized_text = []

        for line in kss.split_sentences(lines):
            line = line.strip()
            sentence_tokenized_text.append(line)

        cleaned_corpus = []

        for sent in sentence_tokenized_text:
            cleaned_corpus.append(self.clean_punc(sent))

        basic_preprocessed_corpus = self.clean_text(cleaned_corpus)

        return basic_preprocessed_corpus


    def clean_punc(self, text):
        for p in self.punct_mapping:
            text = text.replace(p, self.punct_mapping[p])
    
        for p in self.punct:
            text = text.replace(p, f' {p} ')
    
        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
        for s in specials:
            text = text.replace(s, specials[s])
    
        return text.strip()


    def clean_text(self, texts): 
        corpus = [] 
        for i in range(0, len(texts)): 
            review = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',str(texts[i])) #remove punctuation 
            review = re.sub(r'\d+','', str(texts[i]))# remove number 
            review = review.lower() #lower case 
            review = re.sub(r'\s+', ' ', review) #remove extra space 
            review = re.sub(r'<[^>]+>','',review) #remove Html tags 
            review = re.sub(r'\s+', ' ', review) #remove spaces 
            review = re.sub(r"^\s+", '', review) #remove space from start 
            review = re.sub(r'\s+$', '', review) #remove space from the end 
            corpus.append(review) 
                
        return corpus




# khaiii 이용한 품사 구분
class PartOfSpeech:

    def __init__(self):
        self.api = KhaiiiApi() 
        self.significant_tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'VX', 'MAG', 'MAJ', 'XSV', 'XSA']

    def pos_text(self, texts):
        corpus = []
        for sent in texts:
            pos_tagged = ''
            for word in self.api.analyze(sent):
                for morph in word.morphs:
                    if morph.tag in self.significant_tags:
                        pos_tagged += morph.lex + '/' + morph.tag + ' '
            corpus.append(pos_tagged.strip())
        return corpus


    

# Stemming 동사를 원형으로 복원
class Stemming:

    def __init__(self):
        self.p1 = re.compile('[가-힣A-Za-z0-9]+/NN. [가-힣A-Za-z0-9]+/XS.')
        self.p2 = re.compile('[가-힣A-Za-z0-9]+/NN. [가-힣A-Za-z0-9]+/XSA [가-힣A-Za-z0-9]+/VX')
        self.p3 = re.compile('[가-힣A-Za-z0-9]+/VV')
        self.p4 = re.compile('[가-힣A-Za-z0-9]+/VX')

    def stemming_text(self, text):
        corpus = []
        for sent in text:
            ori_sent = sent
            mached_terms = re.findall(self.p1, ori_sent)
            for terms in mached_terms:
                ori_terms = terms
                modi_terms = ''
                for term in terms.split(' '):
                    lemma = term.split('/')[0]
                    tag = term.split('/')[-1]
                    modi_terms += lemma
                modi_terms += '다/VV'
                ori_sent = ori_sent.replace(ori_terms, modi_terms)
        
            mached_terms = re.findall(self.p2, ori_sent)
            for terms in mached_terms:
                ori_terms = terms
                modi_terms = ''
                for term in terms.split(' '):
                    lemma = term.split('/')[0]
                    tag = term.split('/')[-1]
                    if tag != 'VX':
                        modi_terms += lemma
                modi_terms += '다/VV'
                ori_sent = ori_sent.replace(ori_terms, modi_terms)

            mached_terms = re.findall(self.p3, ori_sent)
            for terms in mached_terms:
                ori_terms = terms
                modi_terms = ''
                for term in terms.split(' '):
                    lemma = term.split('/')[0]
                    tag = term.split('/')[-1]
                    modi_terms += lemma
                if '다' != modi_terms[-1]:
                    modi_terms += '다'
                modi_terms += '/VV'
                ori_sent = ori_sent.replace(ori_terms, modi_terms)

            mached_terms = re.findall(self.p4, ori_sent)
            for terms in mached_terms:
                ori_terms = terms
                modi_terms = ''
                for term in terms.split(' '):
                    lemma = term.split('/')[0]
                    tag = term.split('/')[-1]
                    modi_terms += lemma
                if '다' != modi_terms[-1]:
                    modi_terms += '다'
                modi_terms += '/VV'
                ori_sent = ori_sent.replace(ori_terms, modi_terms)
            corpus.append(ori_sent)
        return corpus




# Stopwords 불용어 제거
class Stopwords:

    def __init__(self):
        self.stopwords = ['데/NNB', '좀/MAG', '수/NNB', '등/NNB']

    def remove_stopword_text(self, text):
        corpus = []
        for sent in text:
            modi_sent = []
            for word in sent.split(' '):
                if word not in self.stopwords:
                    modi_sent.append(word)
            corpus.append(' '.join(modi_sent))
        return corpus



