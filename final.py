import re
import pandas as pd 
import numpy as np  
import seaborn as sns
import nltk
from nltk.stem.porter import *
import matplotlib.pyplot as plt
​
train  = pd.read_csv('trainsen.csv')
test = pd.read_csv('testsen.csv')
​
print(train.head())
​
combi = train.append(test, ignore_index=True)
​
​
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 
​
​
​
print('\n\nRemoving  Twitter Handles \n\n')
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
print(combi['tidy_tweet'].head())
​
# remove special characters, numbers, punctuations
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
     
print('\n\nRemoving Short Words\n\n')
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
print(combi['tidy_tweet'].head())
​
​
print('\n\nTweet Tokenization\n\n')
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
print(tokenized_tweet.head())
​
​
"""print('\n\nStemming\n\n')
​
stemmer = PorterStemmer()
​
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
 # stemming
print(tokenized_tweet.head())"""
​
​
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
​
combi['tidy_tweet'] = tokenized_tweet
​
print (combi['tidy_tweet'])
​
​
# All Words
​
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
​
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
​
​
# Non Sexist Words
​
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
​
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
​
#Negative Words
​
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
​
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
​
    return hashtags
​
​
# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])
​
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])
​
# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])
​
​
# Non-Racist tweets
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
​
​
​
# Racits tweets
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
​
​
​
# Bag of Words
"""
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
​
from sklearn.model_selection import train_test_split
​
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]
​
print (train_bow)
​
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=0, test_size=0.3)
"""
​
​
# TF-IDF Vectorizer
​
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
​
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
​
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]
​
​
"""xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]"""
​
# splitting data into training and validation set
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=0, test_size=0.3)
​
​
​
# Xgboost with TF-IDF 94%
​
import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=10, n_estimators=250, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(xtrain_tfidf,ytrain)
​
predictions = xgb_model.predict(xvalid_tfidf)
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
results = confusion_matrix(yvalid, predictions)
print ('Accuracy Score :',accuracy_score(yvalid, predictions))
print ('Report : ')
print (classification_report(yvalid,predictions))
