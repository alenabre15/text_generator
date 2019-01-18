import numpy as np
import tensorflow as tf

from keras_preprocessing import text

from keras import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

with open(r'C:\Users\Vitaly\Documents\text.txt') as f:
    lines = f.read().splitlines()
    
tokenizer = Tokenizer(num_words=None,
                     filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                     lower=False, split=' ')

tokenizer.fit_on_texts(lines)

sequences = tokenizer.texts_to_sequences(lines)

idx_word = tokenizer.index_word

' '.join(idx_word[w] for w in sequences[10][:25])

features = []
labels = []

training_length = 50

for seq in sequences:

    for i in range(training_length, len(seq)):
    
        extract = seq[i - training_length:i + 1]
    
        features.append(extract[:-1])
        labels.append(extract[-1])
    
features = np.array(features)

num_words = len(word_idx) + 1
label_array = np.zeros((len(features), num_words), dtype = np.int8)

for example_index, word_index in enumerate(labels):
    label_array[example_index, word_index] = 1
    
label_array.shape

idx_word[np.argmax(label_array[100])]

model = Sequential()

model.add(
    Embedding(input_dim=num_words,
              input_length=training_length,
              output_dim=100,
              weights=[embedding_matrix],
              trainable=False,
              mask_zero=True))

model.add(Masking(mask_value=0.0))

model.add(LSTM(64, return_sequences=False,
               dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

mode.add(Dense(num_words, activation='softmax'))

model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

glove_vectors = r'C:\Users\Vitaly\Documents\glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None)

vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

word_lookup = {word: vector for word, vector in zip(words, vectors)}

embedding_matrix = np.zeros(num_words, vectors.shape[1])

for i, word in enumerate(word_idx.keys()):
    vector = word.lookup.get(word, None)
    
    if vector is not None:
        embedding_matrix[i + 1, :] = vector

callbacks = [EarlyStopping(monitor='val_loss', patience=5),
             ModelCheckpoint('..models/model/h5'), save_best_only=True,
                             save_weights_only=False]

history = model.fit(X_train, y_train,
                    batch_size=2048, epochs=150,
                    callbacks=callbacks,
                    validation_data=(X_valid, y_valid))

model = load_model('../models/model/h5')
model.evaluate(X_valid, y_valid)
