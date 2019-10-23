import nltk
import numpy
import multiprocessing
from gensim.models import Word2Vec as w2v
from sklearn.datasets import load_files
from preprocesses import remove_stop_words as rsw
from preprocesses import remove_noise_text as rnt
from time import time
import pickle
#-------------------------------------------------------------------------------
with open('s1test_x', 'rb') as test_x:
    s1test_x = pickle.load(test_x)
with open('s1train_x', 'rb') as train_x:
    s1train_x = pickle.load(train_x)

with open('s2test_x', 'rb') as test_x:
    s2test_x = pickle.load(test_x)
with open('s2train_x', 'rb') as train_x:
    s2train_x = pickle.load(train_x)
#-------------------------------------------------------------------------------
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
#-------------------------------------------------------------------------------
t = time()
w2v_model.build_vocab(sentences, progress_per = 10000)
print('Time to build vocab: {} mins'.format(round((time()-t)/60,2)))
t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30,
                report_delay=1.0)
print('Time to train the model: {} mins'.format(round((time()-t)/60,2)))
w2v_model.init_sims(replace = True)
