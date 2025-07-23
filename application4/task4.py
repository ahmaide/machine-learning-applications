import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras import backend as K

df = pd.read_csv('updated_movie_data.csv')
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
x = tfidf.fit_transform(df['review'])
y = df['sentiment'].values

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []

for train_index, val_index in kf.split(x):
    x_train, x_test = x[train_index], x[val_index]
    y_train, y_test = y[train_index], y[val_index]

    model = Sequential([
        Flatten(input_shape=(x_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=BinaryCrossentropy(),
                  metrics=[BinaryAccuracy()])

    model.fit(x_train, y_train, epochs=5, batch_size= 32, validation_split=0.2)

    _, acc = model.evaluate(x_test.toarray(), y_test)
    fold_accuracies.append(acc)
    K.clear_session()

print("Fold Accuracies:", fold_accuracies)
print("Average Accuracy:", np.mean(fold_accuracies))