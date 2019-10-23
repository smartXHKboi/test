import nltk
import numpy as np
import multiprocessing
import gensim
from gensim.models import Word2Vec as w2v
from sklearn.datasets import load_files
from preprocesses import remove_stop_words as rsw
from preprocesses import remove_noise_text as rnt
import pickle
from plotter import tsnescatterplot
#-------------------------------------------------------------------------------
# Load model
with open('w2v_model', 'rb') as mdl:
    w2v_model = pickle.load(mdl)
#-------------------------------------------------------------------------------
# Input some keywords and associated words.
print(w2v_model.wv.index2entity)
print(w2v_model.wv.most_similar(positive=['d']))
#-------------------------------------------------------------------------------
