import pandas as pd

df = pd.read_csv('df_preprocessed.csv')

hun_stopwords = []
with open('stopwords.txt','r') as o:
    for line in o:
        hun_stopwords.append(line)







from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words=hun_stopwords)

ctfidf_model= ClassTfidfTransformer(reduce_frequent_words=True,bm25_weighting=True )
representation_model = KeyBERTInspired()


model = BERTopic(
    language='hungarian',
    #vectorizer_model=vectorizer_model, 
    verbose= True, 
    ctfidf_model=ctfidf_model, 
    representation_model=representation_model,
    n_gram_range=(1,3))

any(df['poem'].isnull())

docs = df['poem']
docs = docs.reset_index(drop = True)

topics, probs = model.fit_transform(docs)

model.get_topic_info()
model.visualize_barchart(n_words=20)

import plotly.express as px

def insert_br_every_fifteen_words(strings):
    result = []
    for string in strings:
        words = string.split()
        formatted = "<br>".join(" ".join(words[i:i+15]) for i in range(0, len(words), 15))
        result.append(formatted)
    return result

model.visualize_documents(docs=insert_br_every_fifteen_words(docs)).write_html('topics_viz.html')

######################################################################################################
################################## EVAL ############################################################
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

documents = model.get_document_info(docs)
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

# Extract vectorizer and analyzer from BERTopic
vectorizer = model.vectorizer_model
analyzer = vectorizer.build_analyzer()

# Extract features for Topic Coherence evaluation
words = vectorizer.get_feature_names_out()
tokens = [analyzer(doc) for doc in cleaned_docs]
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(token) for token in tokens]
topic_words = [[words for words, _ in model.get_topic(topic)] 
        for topic in range(len(set(topics))-1)]

# Evaluate
coherence_model = CoherenceModel(topics=topic_words, 
                            texts=tokens, 
                            corpus=corpus,
                            dictionary=dictionary, 
                            coherence='c_v')
coherence = coherence_model.get_coherence()
coherence_per_topic = coherence_model.get_coherence_per_topic()

print(coherence_per_topic)

docinfo = model.get_document_info(docs)

df_2 = df.join(docinfo[['Topic', 'Probability']])
