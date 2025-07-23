import pyprind 
import pandas as pd
import numpy as np
import os
import sys
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

###################################### 1. Data Preparation ######################################
"""
basepath = 'aclImdb'
labels = {'pos' : 1, 'neg': 0}

pbar = pyprind.ProgBar(5000, stream=sys.stdout)
df = pd.DataFrame()

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)

        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            
            new_row = pd.DataFrame([{'review': txt, 'label': labels[l]}])
            df = pd.concat([df, new_row], ignore_index=True)
        
            pbar.update()

df.columns = ['review', 'sentiment'] 

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv', index=False, encoding='utf-8')

# Clean up data ....
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', '')

    return text

df = pd.read_csv('movie_data.csv', encoding='utf-8', header=None)
df.columns = ['review', 'sentiment']

df['review'] = df['review'].apply(preprocessor)

# Tokenize ...
porter = PorterStemmer()
nltk.download('stopwords')

def tokenizer_porter(text):
    stop = stopwords.words('english')
    return [porter.stem(word) for word in text.split() if word not in stop]

df['review'] = df['review'].apply(tokenizer_porter)
df['review'] = df['review'].apply(lambda tokens: ' '.join(tokens))

df.to_csv('updated_movie_data.csv', index=False, encoding='utf-8')

# Convert to tf-idf vectors ...
df = pd.read_csv('updated_movie_data.csv', encoding='utf-8')

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

X = tfidf.fit_transform(df['review'])
y = df['sentiment'].values

###################################### 2. Building an FNN ######################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate = 0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size = 32, validation_split=0.2)

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: ', accuracy)

"""
###################################### 3. Training a Baseline Model and Tuning the Hyperparameters ######################################
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

df = pd.read_csv('updated_movie_data.csv', encoding='utf-8')
#df = df.sample(n=50000, random_state=42)

tfidf = TfidfVectorizer(strip_accents=None, lowercase=True, stop_words='english')
X = tfidf.fit_transform(df['review'])
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Logistic Regression from the book
class LogisticRegression:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

start_time = time.time()
log_reg = LogisticRegression(eta=0.0001, n_iter=100)
log_reg.fit(X_train.toarray(), y_train)
y_pred_lr = log_reg.predict(X_test.toarray())
lr_time = time.time() - start_time
lr_acc = accuracy_score(y_test, y_pred_lr)

# FNN
model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.00001)),
    Dropout(0.3),
    #Dense(128, activation='relu', kernel_regularizer=l2(0.00001)),
    #Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)), #.01
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

learning_rate = 0.00005
model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
#9
start_time = time.time()
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)
fnn_time = time.time() - start_time

_, fnn_acc = model.evaluate(X_test, y_test)

print(f"Logistic Regression:     {lr_acc:.4f}, Time: {lr_time:.2f} seconds")
print(f"Learning Rate:           {learning_rate:.5f}")
print(f"Tuned FNN:               {fnn_acc:.4f}, Time: {fnn_time:.2f} seconds")