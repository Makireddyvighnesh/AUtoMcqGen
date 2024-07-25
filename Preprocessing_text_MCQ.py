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


import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

stop_words = stopwords.words('english')

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
