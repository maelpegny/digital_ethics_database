# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 07:51:32 2023

@author: maelp
"""

# BERTOPIC clean

# Import necessary packages
from bertopic import BERTopic
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# CODE BLOCK: cleaning and tokenizing text data
# Import database
research_topics = pd.read_csv('research_topics_list.csv')


# Extract the column of research topics in English, drop NaN values and convert
# to list
docs = research_topics['Translated Research Topics'].dropna().to_list()


# Clean out "Not available" values
clean_docs = [x for x in docs if x != 'Not available']


# Clean data from Python symbols
pattern = ('(\\n|\\xa0\\t|\\u200b|\\u202f|\\xad|\\u2028|\\r)')
super_clean_docs = [re.sub(pattern, ' ', x) for x in clean_docs]


# Tokenize words lower-cased data
tokenized_docs = [word_tokenize(doc.lower()) for doc in super_clean_docs]


# Clean data from various signs, stopwords and generic words
stop_words = stopwords.words("english")
stop_words.extend(['include', 'well', 'different', 'especially', 'new'])
extra_signs = ['``', "''", '”', '“', '”', '”', '``', '”', '"']
list_generic_words = ['digital', 'digitalization', 'digitalisation',
                      'ethics', 'ethical', 'political', 'social', 'society',
                      'legal', 'law', 'philosophy', 'sociology', 'studies',
                      'research', 'theory', 'methods', 'questions', 'context',
                      'project', 'analysis', 'areas', 'perspective',
                      'technology', 'technologies', 'artificial',
                      'intelligence', 'ai', 'focuses', 'focus', 'data',
                      'science', 'dissertation', 'interests', 'interested']

super_clean_docs_list = [[element for element in list if element
                          not in stop_words and element not in extra_signs
                          and element not in list_generic_words and element
                          not in string.punctuation]
                         for list in tokenized_docs]


# Convert list of lists of tokens into a list of tokens
super_clean_docs = [element for list in super_clean_docs_list
                    for element in list]



# CODE BLOCK: Bertopic
# Concatenate the document with itself 10 times to augment its size

super_clean_docs_dup = super_clean_docs + super_clean_docs +\
    super_clean_docs + super_clean_docs + super_clean_docs + \
    super_clean_docs + super_clean_docs + super_clean_docs +\
    super_clean_docs + super_clean_docs + super_clean_docs


# Instantiate Bertopic with specific embedding
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", nr_topics=5)


# Fit model and transform data
topics, probs = topic_model.fit_transform(super_clean_docs_dup)


# CODE BLOCK: visualization of Bertopic results
# Perform sanity check on results
topic_model.get_topic_info()

# Look into topic model 0
topic_model.get_topic(0)

# Look at representative docs for topic 0
topic_model.get_representative_docs(0)


# Interactive Intertopic distance visualization of topics
map_chart = topic_model.visualize_topics()
map_chart.show()

# Bar Visualization of main words
bar_chart = topic_model.visualize_barchart(n_words=10)
bar_chart.show()
