#!/usr/bin/env python
# coding: utf-8

# ## Importing the document and extracting text from that using PyPDF2

# In[1]:


import PyPDF2
import re
import string

class PDFCleaner:
    def __init__(self, filename):
        self.filename = filename
        self.all_text = ""

    def extract_text(self, start_page, end_page):
        with open(self.filename, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for i in range(start_page, end_page):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                self.all_text += text
        return self.all_text


    def chunk_text(self, chunk_size):
        chunks = []
        current_chunk = ""
        splitted_text = re.split(r'\. *\n', self.all_text)
        for line in splitted_text:
            if len(current_chunk) + len(line) <= chunk_size:
                current_chunk += line + ".\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + ".\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

# # Example usage
# filename = "./books/andrew-ng-machine-learning-yearning.pdf"
# pdf_cleaner = PDFCleaner(filename)
# all_text = pdf_cleaner.extract_text(5,80)

# chunk_size = 5000
# sentences = pdf_cleaner.chunk_text(chunk_size)

# for i, chunk in enumerate(sentences):
#     print(f"Chunk {i + 1}:")
#     print(len(chunk))
#     print("=" * 20)


# In[2]:


# sentences


# ## Cleaning text and converting into vectors using TF-IDF

# In[3]:


import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

stop_words = stopwords.words('english')

# nltk.download('stopwords')
# nltk.download('wordnet')

class TextSummarizer:
    def __init__(self):
        self.wordnet = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.exclude = set(string.punctuation)
        
    def clean_text(self, text):
        # print(1)
        # print(text)
        all_text = re.sub(r'\n\d+(\.\d+)?', '', str(text))
        all_text = re.sub(r'Figure \d+(\.|\-\d+)?','', all_text)
        all_text = re.sub(r'Chapter \d+(d+)?','',all_text)
        all_text = re.sub(r'Page \d+','', all_text)
        all_text = re.sub(r'--+','', all_text)
        all_text = re.sub(r'ptg[0-9]+', '', all_text)
        all_text = re.sub(r'table', '', all_text, flags=re.IGNORECASE)
        all_text = re.sub(r'https://+', '', all_text)
        all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)
        return all_text

    def preprocess_text(self, text):
        corpus = []
        for i in range(len(text)):
            sentence = self.clean_text(text[i])
            sentence = sentence.lower().split()
            # print(0)
            
            # print(sentence)
              # Remove non-alphabetic characters
            # sentence = cleaned_text.split()
            sentence = [self.wordnet.lemmatize(word) for word in sentence if not word in self.stop_words]
            sentence = [ ch for ch in sentence if ch not in self.exclude]
            
            sentence = ' '.join(sentence)
            # print(sentence)
            corpus.append(sentence)
        return corpus

    def calculate_sentence_scores(self, tfidf_matrix):
        sentence_scores = np.sum(tfidf_matrix, axis=1)  # Sum of TF-IDF scores for each sentence
        return sentence_scores

    def generate_summary(self, document, tfidf_matrix, num_sentences):
        # print(tfidf_matrix, num_sentences)
        sentence_scores = self.calculate_sentence_scores(tfidf_matrix)
        ranked_sentences = sorted(((score, i) for i, score in enumerate(sentence_scores)), reverse=True)
        selected_sentences = sorted([i for _, i in ranked_sentences[:num_sentences]])
        print(selected_sentences)
        summary = ' '.join(document[i] for i in selected_sentences)
        
        return summary

# # Example usage
# text_summarizer = TextSummarizer()

# # Preprocess the text (replace 'sentences' with your actual list of sentences)
# # sentences = [...]  # Your list of sentences here
# preprocessed_corpus = text_summarizer.preprocess_text(sentences)

# # Convert preprocessed text into TF-IDF matrix

# tfidf_vectorizer = TfidfVectorizer()
# print(len(preprocessed_corpus))
# # print(preprocessed_corpus)
# tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_corpus)
# # print(tfidf_matrix)
# num_sentences = int(len(sentences)*0.5)  # Select 25% of sentences as summary
# summary = text_summarizer.generate_summary(document=preprocessed_corpus, tfidf_matrix=tfidf_matrix, num_sentences=num_sentences)
# print("Summary:")
# # print(len(summary))
# print(summary)


# ## Importing Topic modeling algorithms LDA

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import pyLDAvis.gensim_models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

class TopicModeling:
    def __init__(self, sentences):
        self.sentences = sentences
        self.nlp = spacy.load("en_core_web_md")
        self.tokens = []
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.topics = []

    def preprocess_text(self):
        for summary in self.nlp.pipe(self.sentences):
            proj_tok = [token.lemma_.lower() for token in summary if not token.is_stop and token.is_alpha]
            self.tokens.append(proj_tok)

    def build_lda_model(self, num_topics):
       
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, iterations=50, num_topics=num_topics, random_state=100)

    def visualize_coherence(self, start, stop, step):
        self.dictionary = Dictionary(self.tokens)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.tokens]
        topics = []
        score = []
        for i in range(start, stop, step):
            lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, iterations=10, num_topics=i, random_state=100)
            cm = CoherenceModel(model=lda_model, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
            topics.append(i)
            score.append(cm.get_coherence())
        plt.plot(topics, score)
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.show()
        return topics[score.index(min(score))]

    def show_topics(self, num_topics, num_words):
        show_topics = self.lda_model.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=True)
        for topic_id, representation in show_topics:
            print(f"Topic {topic_id}:")
            words = [token.split('"')[1] for token in representation.split() if token.endswith('"')]
            self.topics.append(words)
            print(words[:])
        return self.topics

    def find_most_relevant_topic(self, user_topic):
        max_matching_words = 0
        most_relevant_topic_index = -1
        for i, topic_words in enumerate(self.topics):
            matching_words = sum(word in user_topic for word in topic_words)
            if matching_words > max_matching_words:
                max_matching_words = matching_words
                most_relevant_topic_index = i
        return most_relevant_topic_index

    def find_most_relevant_topics_for_user_topics(self, user_topics):
        lemmatizer = WordNetLemmatizer()
        lemmatized_user_topics = []
        print("user topics", user_topics)
        for topic in user_topics:
            print("topic", topic)
            lemmatized_topic = lemmatizer.lemmatize(topic)
            lemmatized_user_topics.append(lemmatized_topic)
        
        topic_texts = [' '.join(topic_words) for topic_words in self.topics]
        vectorizer = TfidfVectorizer()
        topic_vectors = vectorizer.fit_transform(topic_texts)
        indexes = []
        for i, user_topic in enumerate(lemmatized_user_topics):
            user_topic_text = ' '.join(user_topic)
            user_vector = vectorizer.transform([user_topic_text])
            similarities = cosine_similarity(topic_vectors, user_vector)
            most_similar_topic_index = similarities.argmax()
            indexes.append(most_similar_topic_index)
        print("Lemmatized User topics are: ", lemmatized_user_topics)
        return lemmatized_user_topics,indexes


# # Example usage:
# # sentences = [...]  # Your list of sentences
# model = TopicModeling(sentences)
# model.preprocess_text()
# num_topics = model.visualize_coherence(start=1, stop=20, step=1)
# model.build_lda_model(num_topics)
# topics = model.show_topics(num_topics, num_words=100)
# user_topics = [["variance"], ["bias"], ["machine"], ["learning"], ["metrics"], ["precision"]]
# user_topics, relevant_topics = model.find_most_relevant_topics_for_user_topics(user_topics)
# print(user_topics, relevant_topics)


# In[5]:


# num_topics


# In[6]:


# user_topics[0][0] in ['algorithm', 'set', 'example', 'learning', 'error', 'training', 'datum', 'dev', 'performance', 'machine', 'end', 'image', 'suppose', 'curve', 'ng', 'learn', 'yearning', 'page', 'human', 'achieve', 'score', 'andrew', 'high', 'draft', 'system', 'different', 'distribution', 'try', 'reward', 'bias', 'function', 'large', 'car', 'optimal', 'rate', 'trajectory', 'level', 'reinforcement', 'use', 'work', 'task', 'speech', 'landing', 'output', 'test', 'train', 'draw', 'give', 'input', 'optimization', 'internet', 'label', 'small', 'mobile', 'component', 'problem', 'build', 'cat', 'good', 'variance', 'apply', 'desire', 'recognize', 'neural', 'negative', 'helicopter', 'improve', 'case', 'bad', 'audio', 'network', 'recognition', 'find', 'pipeline', 'search', 'dataset', 'avoidable', 'look', 'hand', 'size', 'choose', 'add', 'app', 'well', 'tell', 'phoneme', 'follow', 'like', 'positive', 'obtain', 'know', 'design', 'subset', 'y', 'low', 'estimate', 'indicate', 'model', 'need', 'measure']


# In[7]:


# len(topics[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[380]:


# Import
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# import spacy
# import pyLDAvis.gensim_models
# pyLDAvis.enable_notebook()# Visualise inside a notebook
# import en_core_web_md
# from gensim.corpora import Dictionary
# from gensim.models import LdaModel
# from gensim.models import CoherenceModel

# # Read the data
# reports = sentences
# # Our spaCy model:
# nlp = en_core_web_md.load()

# Tags I want to remove from the text
# # removal= ['PUNCT','ADP','SPACE', 'NUM', 'SYM']
# tokens = []

# for summary in nlp.pipe(reports):
#    proj_tok = [token.lemma_.lower() for token in summary if not token.is_stop and token.is_alpha]
#    tokens.append(proj_tok)

# print(tokens)


# ## create a dictionary and corpus

# In[381]:


# dictionary = Dictionary(tokens)
# dictionary.token2id


# ## filter out low-frequency and high frequency tokens

# In[382]:


# dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100)


# In[383]:


# corpus = [dictionary.doc2bow(doc) for doc in tokens]
# corpus


# In[384]:


# dictionary.doc2bow(tokens[3])


# ## Model building

# In[385]:


# lda_model = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=10)


# In[406]:


# topics =[]
# score = []
# for i in range(1,20,1):
#     lda_model = LdaModel(corpus = corpus, id2word=dictionary, iterations=10, num_topics=i, random_state=100)
#     cm = CoherenceModel(model = lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
#     topics.append(i)
#     score.append(cm.get_coherence())

# _=plt.plot(topics, score)
# _=plt.xlabel('Number of Topics')
# _=plt.ylabel('Coherence Score')
# plt.show()


# In[476]:


# score.index(min(score))


# In[408]:


# lda_model = LdaModel(corpus=corpus, id2word=dictionary, iterations=100, num_topics=score.index(min(score)))


# In[409]:


# lda_model.print_topics()


# In[410]:


# lda_model[corpus][0]


# In[411]:


# topic = [sorted(lda_model[corpus][text])[0][0] for text in range(len(corpus))]
# show_topics = lda_model.show_topics(num_topics=5, num_words=80, log=False, formatted=True)
# show_topics


# In[412]:


# Print top 50 words for each topic
# topics =[]
# for topic_id, representation in show_topics:
#     print(f"Topic {topic_id}:")
#     words = [token.split('"')[1] for token in representation.split() if token.endswith('"')]
#     topics.append(words)
#     print(words[:])  # Print the first 50 words


# In[413]:


# # User-provided topic
# user_topic = "variance"

# # Function to find the most relevant topic based on keyword matching
# def find_most_relevant_topic(user_topic, topics):
#     max_matching_words = 0
#     most_relevant_topic_index = -1

#     # Iterate over each topic
#     for i, topic_words in enumerate(topics):
#         matching_words = sum(word in user_topic for word in topic_words)
#         # print(matching_words)
#         if matching_words > max_matching_words:
#             max_matching_words = matching_words
#             most_relevant_topic_index = i

#     return most_relevant_topic_index

# # Find the most relevant topic index
# most_relevant_topic_index = find_most_relevant_topic(user_topic, topics = topics)

# # Retrieve the words associated with the most relevant topic
# relevant_words = topics[most_relevant_topic_index]

# print("Most relevant topic:")
# print(most_relevant_topic_index)
# print(relevant_words)


# In[486]:


# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()
# # User-provided topics
# user_topics = [["variance"], ["bias"], ["machine"], ["learning"], ["metrics"], ["precision"]]

# lemmatized_user_topics = []
# for topic in user_topics:
#     lemmatized_topic = [lemmatizer.lemmatize(word) for word in topic]
#     lemmatized_user_topics.append(lemmatized_topic)
    
# # Combine topic words into a single string
# topic_texts = [' '.join(topic_words) for topic_words in topics]

# # Vectorize the LDA topics
# vectorizer = TfidfVectorizer()
# topic_vectors = vectorizer.fit_transform(topic_texts)

# indexes = []
# Vectorize each user topic separately and calculate cosine similarity with LDA topics
# for i, user_topic in enumerate(lemmatized_user_topics):
#     user_topic_text = ' '.join(user_topic)
#     user_vector = vectorizer.transform([user_topic_text])
#     similarities = cosine_similarity(topic_vectors, user_vector)
#     print(similarities)
#     # Find the index of the most similar topic
#     most_similar_topic_index = similarities.argmax()

#     # Retrieve the words associated with the most relevant topic
#     relevant_words = topics[most_similar_topic_index]
#     indexes.append(most_similar_topic_index)
#     print(f"Most relevant topic for user topic {i+1}:")
#     print(most_similar_topic_index)
#     print(relevant_words)


# In[425]:


# lemmatized_user_topics[4][0] in topics[2]


# In[426]:


# user_topics


# In[427]:


# lemmatized_user_topics, indexes


# In[418]:


# similarities


# In[421]:


# len(sentences)


# In[480]:


# user_topics[4][0] in topics[6]


# In[ ]:




