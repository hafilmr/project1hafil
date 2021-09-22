#!/usr/bin/env python
# coding: utf-8

# In[7]:


#IMPORT LIBRARY

import pandas as pd
import numpy as np
import nltk
import string
import re
import csv
import ast
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn import svm
from googletrans import Translator


# In[8]:


#DEKLARASIKAN DATA

def load_data():
    dataset = pd.read_csv('DatasetTweet.csv', encoding = 'unicode_escape')
    return dataset


# In[9]:


df = load_data()


# In[10]:


df.head()


# In[11]:


df


# In[12]:


df.drop(df.columns[[0,1]], axis = 1, inplace = True)


# In[13]:


df


# In[14]:


df = df.dropna()


# In[15]:


df


# In[16]:


df


# In[19]:


df['Tweet'].describe()


# In[20]:


df.info()


# In[21]:


df['Jumlah Kata per Tweet'] = df['Tweet'].str.split().str.len()


# In[22]:


df


# In[23]:


df['Jumlah Kata per Tweet'].describe()


# In[24]:


# df.drop_duplicates(subset = 'Tweet', keep = 'first', inplace=True)


# In[25]:


nltk.download('stopwords')


# In[26]:


from nltk.corpus import stopwords
print(stopwords.words("indonesian"))


# In[27]:


#stopwords
from nltk.corpus import stopwords
stopwords_indonesia = stopwords.words('indonesian')

#stemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#tokenize
from nltk.tokenize import TweetTokenizer


# In[28]:


def casefolding(review):
    review = review.lower()
    return review
 
def tokenize(review):
    token = nltk.word_tokenize(review)
    return token
 
def filtering(review):
    # Remove link web
    review = re.sub(r'http\S+', '', review)
    # Remove @username
    review = re.sub('@[^\s]+', '', review)
    # Remove #tagger
    review = re.sub(r'#([^\s]+)', '', review)
    # Remove angka termasuk angka yang berada dalam string
    # Remove non ASCII chars
    review = re.sub(r'[^\x00-\x7f]', r'', review)
    review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
    review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
    review = re.sub(r'\\u\w\w\w\w', '', review)
    # Remove simbol, angka dan karakter aneh
    review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
    return review
 
def replaceThreeOrMore(review):
    # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", review)
 
# def convertToSlangword(review):
#     kamus_slangword = eval(open("slangwords.txt").read()) # Membuka dictionary slangword
#     pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
#     content = []
#     for kata in review:
#         filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
#         content.append(filteredSlang.lower())
#     review = content
#     return review
 
# def removeStopword(review):
#     stopwords = open(stopwords_Reduced.txt', 'r').read().split()
#     content = []
#     filteredtext = [word for word in review.split() if word not in stopwords]
#     content.append(" ".join(filteredtext))
#     review = content
#     return review


# In[29]:


def convertToSlangword(review):
    kamus_slangword = eval(open("update_combined_slang_words.txt").read()) # Membuka dictionary slangword
    pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
    content = []
    for kata in review:
        filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
        content.append(filteredSlang.lower())
    review = content
    return review


# In[30]:


def removeStopword(review):
    stopwords = open("combined_stop_words.txt", 'r').read().split()
    content = []
    filteredtext = [word for word in review.split() if word not in stopwords]
    content.append(" ".join(filteredtext))
    review = content
    return review


# In[31]:


datasets = [df]
datasets


# In[32]:


for teks in datasets:
    teks = teks['Tweet'].apply(casefolding)
    teks = teks.apply(filtering)
    teks = teks.apply(replaceThreeOrMore)
    teks = teks.apply(tokenize)
    teks = teks.apply(convertToSlangword)
    teks = teks.apply(" ".join)
    teks = teks.apply(removeStopword)
    teks = teks.apply(" ".join)
    print(teks)


# In[33]:


review_dict = {'Tweet': teks}
dfdata = pd.DataFrame(review_dict, columns = ['Tweet'])
print(df.info())
dfdata.to_csv('dataset-bersih_skripsi.csv', sep= ',' , encoding='utf-8')


# In[34]:


dfdata


# In[35]:


dfdata['Jumlah Kata'] = dfdata['Tweet'].str.split().str.len()


# In[36]:


dfdata


# In[37]:


dfdata['Jumlah Kata'].value_counts()


# In[38]:


dfdata.drop_duplicates(subset = 'Tweet', keep = 'first', inplace=True)


# In[39]:


dfdata


# In[40]:


dfdata['Jumlah Kata'].value_counts()


# In[43]:


dfdata


# In[44]:


dfdata.to_csv('DatasetHasilPenelitian.csv', index = False)


# In[46]:


TA_df = pd.read_csv('DatasetHasilPenelitian.csv', encoding = 'unicode_escape')


# In[47]:


TA_ta = TA_df.dropna()


# In[48]:


TA_ta


# In[50]:


TA_ta.to_csv('DatasetHasilPenelitian.csv', index=False)


# In[53]:


TA = pd.read_csv('DatasetHasilPenelitian.csv', encoding = 'unicode_escape')


# In[54]:


TA


# In[55]:


TA


# In[56]:


from nltk.tokenize import word_tokenize


# In[57]:


import nltk
nltk.download('punkt')


# In[58]:


word_dict = {}
for i in range(0,len(TA['Tweet'])):
    sentence = TA['Tweet'][i]
    word_token = word_tokenize(sentence)
    for j in word_token:
        if j not in word_dict:
            word_dict[j] = 1
        else:
            word_dict[j] += 1


# In[59]:


len(word_dict)


# In[60]:


len({k:v for (k,v) in word_dict.items() if v < 4})


# LEXICON

# In[61]:


pos_lexicon = pd.read_csv('InSet-master\positive.tsv',sep='\t')
neg_lexicon = pd.read_csv('InSet-master\negative.tsv',sep='\t')


# In[62]:


pos_lexicon


# In[63]:


neg_lexicon


# In[64]:


posneg_lexicon = pos_lexicon.append(neg_lexicon,ignore_index=True)


# In[65]:


len(posneg_lexicon)
posneg_lexicon


# In[66]:


# addition = pd.read_csv('agusmakmun\sentimentword.csv')


# In[67]:


# len(addition)


# In[68]:


# addition.isnull().sum()


# In[69]:


posneg_lexicon.isnull().sum()


# In[70]:


lexicon_word = posneg_lexicon['word'].to_list()


# In[71]:


posneg_lexicon['word'][0] in lexicon_word


# In[72]:


# add_word = []
# add_weight = []
# for i in range(0,len(addition)):
# #     if (addition['word'][i] not in lexicon_word):
#         add_word.append(addition['word'][i])
#         add_weight.append(addition['weight'][i])

# addition_lexicon = pd.DataFrame(list(zip(add_word,add_weight)),columns =['word','weight'])


# In[73]:


# addition_lexicon


# In[74]:


# next_lexicon = posneg_lexicon.append(addition_lexicon,ignore_index = True)


# In[75]:


# next_lexicon


# In[76]:


# next_lexicon.sample(5)


# In[77]:


# next_lexicon[next_lexicon['weight']<0].min()


# In[78]:


my_file = open("swearwords\swear-words.txt", "r")
content = my_file.read()
swear_words = content.split("\n")


# In[79]:


swear_words


# In[80]:


weight_swear = [-5 for i in range(len(swear_words))]


# In[81]:


swear_lexicon = pd.DataFrame(list(zip(swear_words,weight_swear)),columns =['word','weight'])


# In[82]:


swear_lexicon


# In[83]:


swear_lexicon.to_csv('swear.csv', index = False)


# In[84]:


featswear_lexicon = posneg_lexicon.append(swear_lexicon,ignore_index = True)


# In[85]:


len(featswear_lexicon)


# In[86]:


featswear_lexicon


# In[87]:


def number_of_words(x):
    words = word_tokenize(x['word'])
    number = len(words)
    return number


# In[88]:


featswear_lexicon['number_of_words'] = featswear_lexicon.apply(lambda x: number_of_words(x),axis=1)


# In[89]:


featswear_lexicon = featswear_lexicon.drop(featswear_lexicon[featswear_lexicon['number_of_words'] == 0].index[0],axis=0)


# In[90]:


featswear_lexicon  = featswear_lexicon.reset_index(drop=True)


# In[91]:


featswear_lexicon[featswear_lexicon['number_of_words'] == 3]


# In[92]:


featswear_lexicon.to_csv('featswear_lexicon.csv',index=False)


# In[93]:


featswear_lexicon


# In[94]:


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()


# In[95]:


my_file = open("combined_stop_words.txt", "r")
content = my_file.read()
stop_words = content.split("\n")


# In[96]:


lexicon_word = featswear_lexicon['word'].to_list()


# In[97]:


overlap=[]
for word in stop_words:
    if word in lexicon_word:
        overlap.append(word)
    else:
        kata_dasar = stemmer.stem(word)
        if kata_dasar in lexicon_word:
            overlap.append(word)


# In[98]:


negasi = ['bukan','tidak','ga','gk']
featswear_lexicon= pd.read_csv('featswear_lexicon.csv')
featswear_lexicon = featswear_lexicon.drop(featswear_lexicon[(featswear_lexicon['word'] == 'bukan')
                               |(featswear_lexicon['word'] == 'tidak')
                               |(featswear_lexicon['word'] == 'ga')|(featswear_lexicon['word'] == 'gk') ].index,axis=0)
featswear_lexicon = featswear_lexicon.reset_index(drop=True)


# In[99]:


len(featswear_lexicon)


# In[100]:


featswear_lexicon.head()


# In[101]:


# Lexicon_Based = featswear_lexicon.append(modified_lexicon, ignore_index = True)


# In[102]:


# Lexicon_Based


# In[103]:


featswear_lexicon.to_csv('featswear_lexicon.csv', index=False)


# In[104]:


Lexicon_Based = pd.read_csv('featswear_lexicon.csv', encoding = "unicode_escape")


# In[105]:


# Lexicon_Based = Lexicon_Based.drop_duplicates(subset = 'word', keep = 'first')


# In[106]:


Lexicon_Based


# In[107]:


lexicon_word = Lexicon_Based['word'].to_list()
lexicon_num_words = Lexicon_Based['number_of_words']


# In[108]:


len(lexicon_word)


# In[109]:


ns_words = []
factory = StemmerFactory()
stemmer = factory.create_stemmer()
for word in word_dict.keys():
    if word not in lexicon_word:
        kata_dasar = stemmer.stem(word)
        if kata_dasar not in lexicon_word:
            ns_words.append(word)
len(ns_words)


# In[110]:


len({k:v for (k,v) in word_dict.items() if ((k in ns_words)&(v>3)) })


# In[111]:


ns_words_list = {k:v for (k,v) in word_dict.items() if ((k in ns_words)&(v>3))}


# In[112]:


sort_orders = sorted(ns_words_list.items(), key=lambda x: x[1], reverse=True)
sort_orders=sort_orders[0:20]
for i in sort_orders:
    print(i[0], i[1])


# In[113]:


def del_word(x,key_list):
    n = len(key_list)
    word_tokens = word_tokenize(x)
    new_x =''
    for word in word_tokens:
        if word not in key_list:
            new_x = new_x+word+' '
    return new_x


# In[114]:


word_to_plot = TA['Tweet'].copy()


# In[115]:


word_to_plot_1 = word_to_plot.apply(lambda x: del_word(x,negasi))


# In[116]:


wordcloud = WordCloud(width = 800, height = 800, background_color = 'white', max_words = 100
                      , min_font_size = 20).generate(str(word_to_plot_1))
#plot the word cloud
fig = plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# SENTIMENT

# In[117]:


'pekerti' in word_dict


# In[118]:


Lexicon_Based['number_of_words'].value_counts()


# In[119]:


'budi baik' in lexicon_word


# #PROSES PELABELAN

# In[120]:


sencol =[]
senrow =np.array([])
nsen = 0
factory = StemmerFactory()
stemmer = factory.create_stemmer()
sentiment_list = []
# function to write the word's sentiment if it is founded
def found_word(ind,words,word,sen,sencol,sentiment,add):
    # if it is already included in the bag of words matrix, then just increase the value
    if word in sencol:
        sen[sencol.index(word)] += 1
    else:
    #if not, than add new word
        sencol.append(word)
        sen.append(1)
        add += 1
    #if there is a negation word before it, the sentiment would be the negation of it's sentiment
    if (words[ind-1] in negasi):
        sentiment += -Lexicon_Based['weight'][lexicon_word.index(word)]
    else:
        sentiment += Lexicon_Based['weight'][lexicon_word.index(word)]
    
    return sen,sencol,sentiment,add


# In[121]:


# checking every words, if they are appear in the lexicon, and then calculate their sentiment if they do
for i in range(len(TA)):
    nsen = senrow.shape[0]
    words = word_tokenize(TA['Tweet'][i])
    sentiment = 0 
    add = 0
    prev = [0 for ii in range(len(words))]
    n_words = len(words)
    if len(sencol)>0:
        sen =[0 for j in range(len(sencol))]
    else:
        sen =[]
    
    for word in words:
        ind = words.index(word)
        # check whether they are included in the lexicon
        if word in lexicon_word :
            sen,sencol,sentiment,add= found_word(ind,words,word,sen,sencol,sentiment,add)
        else:
        # if not, then check the root word
            kata_dasar = stemmer.stem(word)
            if kata_dasar in lexicon_word:
                sen,sencol,sentiment,add= found_word(ind,words,kata_dasar,sen,sencol,sentiment,add)
        # if still negative, try to match the combination of words with the adjacent words
            elif(n_words>1):
                if ind-1>-1:
                    back_1    = words[ind-1]+' '+word
                    if (back_1 in lexicon_word):
                        sen,sencol,sentiment,add= found_word(ind,words,back_1,sen,sencol,sentiment,add)
                    elif(ind-2>-1):
                        back_2    = words[ind-2]+' '+back_1
                        if back_2 in lexicon_word:
                            sen,sencol,sentiment,add= found_word(ind,words,back_2,sen,sencol,sentiment,add)
    # if there is new word founded, then expand the matrix
    if add>0:  
        if i>0:
            if (nsen==0):
                senrow = np.zeros([i,add],dtype=int)
            elif(i!=nsen):
                padding_h = np.zeros([nsen,add],dtype=int)
                senrow = np.hstack((senrow,padding_h))
                padding_v = np.zeros([(i-nsen),senrow.shape[1]],dtype=int)
                senrow = np.vstack((senrow,padding_v))
            else:
                padding =np.zeros([nsen,add],dtype=int)
                senrow = np.hstack((senrow,padding))
            senrow = np.vstack((senrow,sen))
        if i==0:
            senrow = np.array(sen).reshape(1,len(sen))
    # if there isn't then just update the old matrix
    elif(nsen>0):
        senrow = np.vstack((senrow,sen))
        
    sentiment_list.append(sentiment)


# In[122]:


len(sentiment_list)


# In[123]:


print(senrow.shape[0])


# In[124]:


sencol.append('BOBOT TWEET')
sentiment_array = np.array(sentiment_list).reshape(senrow.shape[0],1)
sentiment_data = np.hstack((senrow,sentiment_array))
TA_sen = pd.DataFrame(sentiment_data,columns = sencol)


# In[125]:


TA_sen.head(638)


# In[126]:


cek_ta = pd.DataFrame([])
cek_ta['Tweet'] = TA['Tweet'].copy()
cek_ta['Label Tweet']  = TA_sen['BOBOT TWEET'].copy()


# In[127]:


cek_ta.sort_values('Label Tweet')


# In[128]:


import seaborn as sns
sns.set(style="whitegrid", palette="colorblind", color_codes=True)
sns.kdeplot(cek_ta['Label Tweet'],color='g',shade=True)
plt.title('Distribusi Pelabelan')
plt.xlabel('Nilai Label')


# In[129]:


sns.set(style="whitegrid")
sns.boxplot(x=cek_ta['Label Tweet'])


# In[130]:


cek_ta['Label Tweet'].value_counts()


# In[131]:


cek_ta.to_csv('HasilLexiconRevisi.csv', index = False)


# In[ ]:




