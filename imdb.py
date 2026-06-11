# DL.-imdb-
model prediction on imdb with tensorflow
## CODE


from tensorflow.keras.datasets import imdb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=100)
x_test  = pad_sequences(x_test, maxlen=100)

model = Sequential([ Embedding(10000, 16, input_length=100),Flatten(), Dense(64, activation='relu'),Dense(2, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train, y_train, epochs=5, batch_size=256)
model.evaluate(x_test, y_test)
print("predictions =", model.predict(x_test[0:5]))
