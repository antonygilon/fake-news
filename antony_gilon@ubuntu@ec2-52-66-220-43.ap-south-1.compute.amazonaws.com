{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd, numpy as np, re, time\n",
    "from nltk.stem.porter import PorterStemmer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.svm import LinearSVC\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading data from json file\n",
    "data = pd.read_json(\"Sarcasm_Headlines_Dataset.json\", lines = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>article_link</th>\n",
       "      <th>headline</th>\n",
       "      <th>is_sarcastic</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>https://www.huffingtonpost.com/entry/versace-b...</td>\n",
       "      <td>former versace store clerk sues over secret 'b...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>https://www.huffingtonpost.com/entry/roseanne-...</td>\n",
       "      <td>the 'roseanne' revival catches up to our thorn...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>https://local.theonion.com/mom-starting-to-fea...</td>\n",
       "      <td>mom starting to fear son's web series closest ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>https://politics.theonion.com/boehner-just-wan...</td>\n",
       "      <td>boehner just wants wife to listen, not come up...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>https://www.huffingtonpost.com/entry/jk-rowlin...</td>\n",
       "      <td>j.k. rowling wishes snape happy birthday in th...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                        article_link  \\\n",
       "0  https://www.huffingtonpost.com/entry/versace-b...   \n",
       "1  https://www.huffingtonpost.com/entry/roseanne-...   \n",
       "2  https://local.theonion.com/mom-starting-to-fea...   \n",
       "3  https://politics.theonion.com/boehner-just-wan...   \n",
       "4  https://www.huffingtonpost.com/entry/jk-rowlin...   \n",
       "\n",
       "                                            headline  is_sarcastic  \n",
       "0  former versace store clerk sues over secret 'b...             0  \n",
       "1  the 'roseanne' revival catches up to our thorn...             0  \n",
       "2  mom starting to fear son's web series closest ...             1  \n",
       "3  boehner just wants wife to listen, not come up...             1  \n",
       "4  j.k. rowling wishes snape happy birthday in th...             0  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "article_link    False\n",
      "headline        False\n",
      "is_sarcastic    False\n",
      "dtype: bool\n"
     ]
    }
   ],
   "source": [
    "print(data.isnull().any(axis = 0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Relacing special symbols and digits in headline column\n",
    "# re stands for Regular Expression\n",
    "data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "# getting features and labels\n",
    "features = data['headline']\n",
    "labels = data['is_sarcastic']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Stemming our data\n",
    "ps = PorterStemmer()\n",
    "features = features.apply(lambda x: x.split())\n",
    "features = features.apply(lambda x :\" \".join([ps.stem(word) for word in x]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# vectorizing the data with maximum of 5000 features\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "tv = TfidfVectorizer(max_features = 5000)\n",
    "features = list(features)\n",
    "features = tv.fit_transform(features).toarray()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "# getting training and testing data\n",
    "features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .05, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9093524612777362\n",
      "0.8375748502994012\n"
     ]
    }
   ],
   "source": [
    "lsvc = LinearSVC()\n",
    "# training the model\n",
    "lsvc.fit(features_train, labels_train)\n",
    "# getting the score of train and test data\n",
    "print(lsvc.score(features_train, labels_train)) # 90.93\n",
    "print(lsvc.score(features_test, labels_test))   # 83.75"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'features_train' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-1-9ea276bff081>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mxgboost\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mxgb\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0mxgb_model\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mxgb\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mXGBClassifier\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmax_depth\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mn_estimators\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m250\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlearning_rate\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcolsample_bytree\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m.7\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgamma\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreg_alpha\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m4\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobjective\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'binary:logistic'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0meta\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.3\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msilent\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0msubsample\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0.8\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfeatures_train\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mlabels_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mpredictions\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mxgb_model\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfeatures_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmetrics\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mconfusion_matrix\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'features_train' is not defined"
     ]
    }
   ],
   "source": [
    "import xgboost as xgb\n",
    "xgb_model = xgb.XGBClassifier(max_depth=10, n_estimators=250, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(features_train,labels_train)\n",
    "\n",
    "predictions = xgb_model.predict(features_test)\n",
    "from sklearn.metrics import confusion_matrix \n",
    "from sklearn.metrics import accuracy_score \n",
    "from sklearn.metrics import classification_report \n",
    "results = confusion_matrix(labels_test, predictions)\n",
    "print ('Accuracy Score :',accuracy_score(labels_test, predictions))\n",
    "print ('Report : ')\n",
    "print (classification_report(labels_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "acmenv",
   "language": "python",
   "name": "acmenv"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
