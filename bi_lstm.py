from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Bidirectional, LSTM, GlobalMaxPool1D,Conv1D
from keras.layers import MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.layers import Add, Concatenate, Subtract, Multiply, Dot, Average
from keras.layers import concatenate
from keras.layers import Flatten, Dropout, Input
from keras.layers import Embedding
from keras.utils import to_categorical
import os, re, csv, math, codecs
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
from string import digits


np.random.seed(121)
# define documents

# to do --> drop single letters 

def load_data(loc):
	#docs = pd.read_csv(loc, delimiter="\t", header=None, encoding='latin-1')
	docs = [line.rstrip('\n') for line in open(loc)]
	docs = [line.lower() for line in [word.replace("\xad", "").replace(".", "").replace("/", "").replace("-", "").replace(",", "").replace("?", "").replace(")", "").replace("(", "").replace("'", " ").replace('"', '') for word in docs]]
	docs = [re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', word) for word in docs]
	return docs

data_en = load_data('/u/bhardwas/Documents/NLP2/data/tp2/train.en')
data_fr = load_data('/u/bhardwas/Documents/NLP2/data/tp2/train.fr')

test_en = load_data('/u/bhardwas/Documents/NLP2/data/tp2/test.en')
test_fr = load_data('/u/bhardwas/Documents/NLP2/data/tp2/test.fr')
	
# define class labels
labels = pd.read_csv('/u/bhardwas/Documents/NLP2/data/tp2/train.y', delimiter="\t", header=None)
target = pd.read_csv('/u/bhardwas/Documents/NLP2/data/tp2/test.y', delimiter="\t", header=None)

# Count Layer
def cnt_feat(data):
	count = []
	for line in data:
		l = len(line.split())
		count.append(l)
	return count

data_en_cnt = cnt_feat(data_en)
data_fr_cnt = cnt_feat(data_fr)

test_en_cnt = cnt_feat(test_en)
test_fr_cnt = cnt_feat(test_fr)

# Translate Layer

trans = pd.read_csv('/u/bhardwas/Documents/NLP2/data/tp2/lexique.en-fr', delimiter="\t", header=None)
trans.columns =  ['en','fr']
mydict = dict(zip(trans['en'], trans['fr']))


def translate(data):
	en_fr = []
	for line in data:
		text = []
		for word in line.split():
			if word in mydict:
				x = word.replace(word,str(mydict[word]))
				text.append(x)
		text = ' '.join(text)
		en_fr.append(text)
	return en_fr

data_trans_to_fr = translate(data_en)
test_trans_to_fr = translate(test_en)
			
max_words = 40

def prepros(data, max_length):
	tok = Tokenizer() #lower=True
	tok.fit_on_texts(data)
	# integer encode the documents
	encoded_docs = tok.texts_to_sequences(data)
	# pad documents to a max length of max_words words
	pad_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	return pad_docs, tok

pre_train_en = prepros(data_en, max_words)
padded_docs, t = pre_train_en[0], pre_train_en[1]
vocab_size = len(t.word_index) + 1

pre_train_fr = prepros(data_fr, max_words)
padded_docs_fr, t_fr = pre_train_fr[0], pre_train_fr[1]
vocab_size_fr = len(t_fr.word_index) + 1

pre_test_en = prepros(test_en, max_words)
padded_test, t_test = pre_test_en[0], pre_test_en[1]

pre_test_fr = prepros(test_fr, max_words)
padded_test_fr, t_test_fr = pre_test_fr[0], pre_test_fr[1]

t_pre_train_fr = prepros(data_trans_to_fr, max_words)
t_padded_docs_fr, t_t_fr = t_pre_train_fr[0], t_pre_train_fr[1]
t_vocab_size_fr = len(t_t_fr.word_index) + 1

t_pre_test_fr = prepros(test_trans_to_fr, max_words)
t_padded_test_fr, t_t_test_fr = t_pre_test_fr[0], t_pre_test_fr[1]

# load the whole embedding into memory

def loadEmbd(loc):
	print('loading word embeddings...')
	embeddings_index = {}
	f = codecs.open(loc, encoding='utf-8')
	for line in tqdm(f):
		values = line.rstrip().rsplit(' ')
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('found %s word vectors' % len(embeddings_index))
	return embeddings_index

	
def emb_matrix(t, embeddings_index,vs):
	embedding_matrix = zeros((vs, 300))
	for word, i in t.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			if embedding_vector != []:
				embedding_matrix[i] = embedding_vector
		else:
			embedding_matrix[i] = '100'
	return embedding_matrix

embeddings_index = loadEmbd('/u/bhardwas/Documents/NLP2/data/wiki.en.vec')
embedding_matrix = emb_matrix(t, embeddings_index ,vocab_size)

embeddings_index_fr = loadEmbd('/u/bhardwas/Documents/NLP2/data/wiki.fr.vec')
embedding_matrix_fr = emb_matrix(t_fr, embeddings_index_fr, vocab_size_fr)

# translation layer
trans_embedding_matrix_fr = emb_matrix(t_t_fr, embeddings_index_fr, t_vocab_size_fr)

# embedding layer
a =  Input(shape=(max_words,),dtype='int32',name='a')
b =  Input(shape=(max_words,),dtype='int32',name='b')

# Count Layer
meta_input = Input(shape=(2,), name='meta_input')
num_embd = np.column_stack((data_en_cnt , data_fr_cnt))
num_embd_t = np.column_stack((test_en_cnt , test_fr_cnt))

# Translate Layer
trans_input = Input(shape=(max_words,),dtype='int32',name='trans_input')
ef = Embedding(t_vocab_size_fr, 300, weights=[trans_embedding_matrix_fr], input_length=max_words, trainable=False)(trans_input)

e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_words, trainable=False)(a)
f = Embedding(vocab_size_fr, 300, weights=[embedding_matrix_fr], input_length=max_words, trainable=False)(b)

merge = (Average()([e, f, ef]))

x = merge
x = Bidirectional(LSTM(10, input_shape=(max_words, 300), dropout=0.5))(x)

#x = Bidirectional(LSTM(80, input_shape=(max_words, 300), return_sequences=True, dropout=0.5))(x)
#x = Bidirectional(LSTM(80, return_sequences=True, dropout=0.5))(x)
#x = Bidirectional(LSTM(80,  dropout=0.5))(x)

x = concatenate([x, meta_input])
x = Dense(32, activation="relu")(x)
x = Dropout(0.50)(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.50)(x)
x = Dense(4, activation="relu")(x)
x = Dropout(0.50)(x)
x = Dense(2, activation="relu")(x)
x = Dropout(0.50)(x)

x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=[a, b, trans_input, meta_input] , outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit([padded_docs, padded_docs_fr, t_padded_docs_fr, num_embd], labels[0], batch_size=400, epochs=2)
loss, accuracy = model.evaluate([padded_test, padded_test_fr, t_padded_test_fr, num_embd_t], target[0], verbose=1)
print('Accuracy: %f' % (accuracy*100))


# Predicting the Test set results
y_pred = model.predict([padded_test, padded_test_fr, t_padded_test_fr, num_embd_t])
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target[0], y_pred)
print(cm)


