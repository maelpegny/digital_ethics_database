# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 07:47:00 2023

@author: maelp
"""


# LATENT DIRICHLET ALLOCATION LDA clean 

# Import generic packages for data analysis and vizualisation

import pyLDAvis.gensim_models
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
import pyLDAvis
import gensim
from nltk.corpus import stopwords
import re
import pandas as pd
from pprint import pprint

# Import NLP packages
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# CODE BLOCK: cleaning and tokenizing text data

# Download csv database
database = pd.read_csv('filne_name.csv')

# Select column containing self-description and convert it to list
research_topics_list = database['Translated Research Topics'].dropna(
).to_list()

# Clean data from Python symbols
pattern = ('(\\n|\\xa0\\t|\\u200b|\\u202f|\\xad|\\u2028|\\r)')
research_topics_list_clean = [re.sub(pattern, ' ', list) for list in research_topics_list]


# Tokenize words lower-cased data
tokenized_topics = [word_tokenize(doc.lower())
                    for doc in research_topics_list_clean]

# Clean data from various signs, stopwords and generic words
stop_words = stopwords.words("english")
stop_words.extend(['include', 'well', 'different', 'especially', 'new'])
extra_signs = ['``', "''", '”', '“', '”', '”', '``', '”', '"']
list_generic_words = ['digital', 'digitalization', 'digitalisation', 'ethics',
                      'ethical', 'political', 'social', 'society', 'legal',
                      'law', 'philosophy', 'sociology', 'studies', 'research',
                      'theory', 'methods',
                      'questions', 'context', 'project', 'analysis', 'areas',
                      'perspective', 'technology', 'technologies',
                      'artificial', 'intelligence', 'ai', 'focuses', 'focus',
                      'data', 'science', 'dissertation', 'interests',
                      'interested']

tokenized_topics_clean = [
    [element for element in list if element not in stop_words
     and element not in extra_signs and element not in list_generic_words
     and element not in string.punctuation] for list in tokenized_topics]

# CODE BLOCK: formation of gensim corpus

# Convert list of lists of tokens into a list of tokens
token_list = [element  for list in tokenized_topics_clean for element in list]


# Attribute an id to each doc in the list of lists of tokens
dictionary = Dictionary(tokenized_topics_clean)

# Formation of a  gensim corpus
corpus = [dictionary.doc2bow(doc) for doc in tokenized_topics_clean]


# CODE BLOCK Latent Dirichlet Allocation

# Fix the parameter "number of topics"
num_topics = 5

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics)

# Print topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Visualization of LDA results with pyLDAvis package
pyLDAvis.enable_notebook()
pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
