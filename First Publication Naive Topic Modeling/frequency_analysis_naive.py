# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 07:49:04 2023

@author: maelp
"""

# FREQUENCY ANALYSIS AND WORDCLOUD clean 

# Import generic packages for data analysis and vizualisation
import pandas as pd
import matplotlib.pyplot as plt

# Import NLP packages
from wordcloud import WordCloud
from gensim.corpora.dictionary import Dictionary
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


# CODE BLOCK: clean and tokenize text data
# Download csv database
data = pd.read_csv('file_name.csv')


# Select column containing self-description and convert it to list
research_topics_list = data['Translated Research Topics'].dropna().to_list()

# Clean data from Python symbols
pattern = ('(\\n|\\xa0\\t|\\u200b|\\u202f|\\xad|\\u2028|\\r)')
research_topics_list_clean = [re.sub(pattern, ' ', list) for list in research_topics_list]

# Tokenize words in lower-cased data
tokenized_topics = [word_tokenize(doc.lower())
                    for doc in research_topics_list_clean]


# Clean data from punctuation signs, stopwords and generic words
stop_words = stopwords.words("english")
stop_words.extend(['include', 'well', 'different', 'especially', 'new'])
extra_signs = ['``', "''", '”', '“', '”', '”', '``', '”', '"']
list_generic_words = ['digital', 'digitalization', 'digitalisation',
                      'ethics', 'ethical', 'political', 'social',
                      'society', 'legal', 'law', 'philosophy',
                      'sociology', 'studies', 'research', 'theory', 'methods',
                      'questions', 'context', 'project', 'analysis', 'areas',
                      'perspective', 'technology', 'technologies',
                      'artificial', 'intelligence', 'ai', 'focuses', 'focus',
                      'data', 'science', 'dissertation',
                      'interests', 'interested']


tokenized_topics_clean = [[element for element in list if element
                           not in stop_words and element
                           not in list_generic_words and element
                           not in string.punctuation and element
                           not in extra_signs] for list in tokenized_topics]


# Convert list of lists of tokens into a list of tokens
token_list = [element for list in tokenized_topics_clean for element in list]

# CODE BLOCK: frequency analysis and word cloud visualization
# Compute and map most frequent tokens in data
token_frequency = nltk.FreqDist(token_list)
print(token_frequency)
token_frequency.plot(20)
plt.show()

# Attribute an id to each doc in the list of lists of tokens           
dictionary = Dictionary(tokenized_topics_clean)


# Formation of a  gensim corpus
corpus = [dictionary.doc2bow(doc) for doc in tokenized_topics_clean]


# WordCloud visualization of most frequent words in research topics
wordcloud = WordCloud(max_words=40, background_color='white').fit_words(
    token_frequency)


# Display the generated image:
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
