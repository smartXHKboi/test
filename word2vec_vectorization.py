import nltk
import numpy
import multiprocessing
from gensim.models import Word2Vec as w2v
from sklearn.datasets import load_files
from preprocesses import remove_stop_words as rsw
from preprocesses import remove_noise_text as rnt
##---------------------Read slice I---------------------------------------------
s1_train = load_files(r'Email Data/Slice I/Train')
s1train_x, s1train_y = s1_train.data, s1_train.target
s1train_y = ['ham' if i == 0 else 'spam' for i in s1train_y]
s1_test = load_files(r'Email Data/Slice I/Test')
s1test_x, s1test_y = s1_test.data, s1_test.target
s1test_y = ['ham' if i == 0 else 'spam' for i in s1test_y]
##---------------------Read slice II--------------------------------------------
s2_train = load_files(r'Email Data/Slice II/Train')
s2train_x, s2train_y = s2_train.data, s2_train.target
s2train_y = ['ham' if i == 0 else 'spam' for i in s2train_y]
s2_test = load_files(r'Email Data/Slice II/Test')
s2test_x, s2test_y = s2_test.data, s2_test.target
s2test_y = ['ham' if i == 0 else 'spam' for i in s2test_y]
##---------------------Preprocess text------------------------------------------
s1train_x = rnt(s1train_x)
s1train_x = rsw(s1train_x)
s1test_x = rnt(s1test_x)
s1test_x = rsw(s1test_x)

s2train_x = rnt(s2train_x)
s2train_x = rsw(s2train_x)
s2test_x = rnt(s2test_x)
s2test_x = rsw(s2test_x)
##---------------------Train the w2v model--------------------------------------
from time import time
cores = multiprocessing.cpu_count()
w2v_model = w2v(min_count=10,
                window=10,
                size=300,
                sample=1e-5,
                alpha=0.03,
                min_alpha=0.0007,
                negative=5,
                workers=cores-1)
sentences = s1train_x + s2train_x
print(sentences)
t = time()
w2v_model.build_vocab(sentences, progress_per = 10000)
print('Time to build vocab: {} mins'.format(round((time()-t)/60,2)))
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30,
                report_delay=1.0)
print('Time to train the model: {} mins'.format(round((time()-t)/60,2)))
w2v_model.init_sims(replace = True)
import pickle
with open('w2v_model', 'wb') as picklefile:
    pickle.dump(w2v_model, picklefile)
# ##---------------------Vectorize words from datasets----------------------------
#
# ##----------------------Save dataset--------------------------------------------
with open('s1test_x', 'wb') as picklefile:
    pickle.dump(s1test_x, picklefile)
with open('s1test_y', 'wb') as picklefile:
    pickle.dump(s1test_y, picklefile)

with open('s1train_x', 'wb') as picklefile:
    pickle.dump(s1train_x, picklefile)
with open('s1train_y', 'wb') as picklefile:
    pickle.dump(s1train_y, picklefile)

with open('s2test_x', 'wb') as picklefile:
    pickle.dump(s2test_x, picklefile)
with open('s2test_y', 'wb') as picklefile:
    pickle.dump(s2test_y, picklefile)

with open('s2train_x', 'wb') as picklefile:
    pickle.dump(s2train_x, picklefile)
with open('s2train_y', 'wb') as picklefile:
    pickle.dump(s2train_y, picklefile)
