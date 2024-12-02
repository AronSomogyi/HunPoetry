import pandas as pd
from tqdm import tqdm


df = pd.read_csv('df_preprocessed.csv')
df = df.drop(df.columns[:2], axis = 1)

import ast
df['poem'] = df['poem'].apply(ast.literal_eval)
df['poem_lemma'] = df['poem_lemma'].apply(ast.literal_eval)

def join_poem(row):
    return " ".join([word for word in row])

df['poem_lemma'] = df['poem_lemma'].apply(join_poem)

hun_stopwords = []
with open('stopwords.txt','r') as o:
    for line in o:
        hun_stopwords.append(line)
import re
hun_stopwords = [re.sub("\n", "", x) for x in hun_stopwords]

[hun_stopwords.append(x) for x in 
    ['hogyha', 'szó', 'mond', 'akar', 'oly', 'beszéd', 
    'monda', 'miként', 'beszél', 'féle', 'vajha', 'amiképp',
    'vesz', 'tesz', 'dal', 'szem']]

if False:
    # Lemma with huSpacy
    #if True:
    
    import spacy

    nlp = spacy.load("hu_core_news_lg")

    nostop = []
    for rows in tqdm(df['poem']):
        words = [word for word in rows if word not in hun_stopwords]
        nostop.append(words)


    def lemmatize0(tokens):
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
        #return doc

    df['poem_lemma'] = nostop
    tqdm.pandas(desc="Lemmatizing poems")
    df['poem_lemma'] = df['poem_lemma'].progress_apply(lemmatize0)
    
    df.to_csv('df_preprocessed.csv')

docs = df['poem_lemma']
docs = docs.reset_index(drop = True)
docs = list(docs)

# Creating Embedding Model
import numpy as np
#import spacy
from umap import UMAP


if False:
    nlp = spacy.load('hu_core_news_lg')
    nlp.disable_pipes("tagger", "parser", "ner")
    embeddings = np.array([doc.vector for doc in tqdm(nlp.pipe(docs, batch_size=50))])
    np.save("embeddings.npy", embeddings)
embeddings = np.load('embeddings.npy')

umap_model1 = UMAP(n_components=5, n_neighbors=15, min_dist=0.02, random_state= 42)
reduced_embeddings = umap_model1.fit_transform(embeddings)


from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.backend import BaseEmbedder
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=70, min_samples= 8, metric='euclidean', cluster_selection_method='eom', prediction_data=True)


vectorizer_model = CountVectorizer(stop_words=hun_stopwords,ngram_range=(1,2))

ctfidf_model= ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=False)
representation_model = KeyBERTInspired()

class CustomEmbed(BaseEmbedder):
    def __init__(self):
        pass

    def embed(self, documents):
        return reduced_embeddings

empty_umap = BaseDimensionalityReduction()


model = BERTopic(
    verbose=True, 
    embedding_model= CustomEmbed,
    hdbscan_model = hdbscan_model,
    language='hungarian',
    #umap_model=umap_model1,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model, 
    representation_model=representation_model,
    n_gram_range=(1,2))




topics, probs = model.fit_transform(docs, reduced_embeddings)
#topics, probs = model.fit_transform(docs)

model.get_topic_info()

import plotly.express as px
def insert_br_every_fifteen_words(strings):
    result = []
    for string in strings:
        words = string.split()
        formatted = "<br>".join(" ".join(words[i:i+15]) for i in range(0, len(words), 15))
        result.append(formatted)
    return result

docs2 = [
    f"{df['poet'][x]}: {df['Cim'][x]}" + "<br>" + (" ".join(df['poem'][x])) +'<br>' f"Topic: {model.topics_[x]}"
    for x in range(len(df))
]

umap2d_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state = 1).fit_transform(reduced_embeddings)

model.visualize_documents(
    docs=insert_br_every_fifteen_words(docs2), 
    reduced_embeddings=umap2d_embeddings,
    hide_annotations=True,
    topics=list(range(16))).write_html('topics_viz.html')

viz_df = pd.DataFrame({
    'topic' : model.get_document_info(docs).iloc[:,1],
    'x' : umap2d_embeddings[:,0],
    'y' : umap2d_embeddings[:,1],
    'poem' :  insert_br_every_fifteen_words(docs2),
    'poet' : df['poet'],
    'title' : df['Cim']
    #'doc' : docs
})

# topic_names = {key : value for key, value in enumerate([
#             f"{topic}_" + "_".join([word for word, value in model.get_topic(topic)][:3])
#             for topic in set(viz_df['topic'])
#         ])}
# topic_names[-1] = topic_names.pop(19)
# viz_df['names'] = viz_df['topic'].map(topic_names)

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from umap import UMAP
from typing import List, Union

#viz_df = viz_df[~viz_df['topic'].isin([-1])]

def visualize_documents(
    df,
    topics: List[int] = None,
    embeddings: np.ndarray = None,
    reduced_embeddings = umap2d_embeddings,
    sample: float = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Documents and Topics</b>",
    width: int = 1200,
    height: int = 750,
    poet = None,
    poem = None
):



    topic_per_doc = list(df['topic'])
    
    docs = list(df['poem'])

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = min(len(s), int(len(s) * sample))
        if size == 0:
            continue  # Skip topics with no documents after filtering
        indices.extend(np.random.choice(s, size=size, replace=False))
        #print(indices)
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]
    

    #df['names2'] = [viz_df.names[index] for index in indices]
    
    # df = pd.DataFrame({"topic": np.array(topic_per_doc)})
    # df["doc"] = docs
    #df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine").fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]

    unique_topics = set(df['topic'])
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]
    df['poet'] = [viz_df.poet[index] for index in indices]
    df['title'] = [viz_df.title[index] for index in indices]
   
    
    # Prepare text and names 
    # EZ SZEDI KI A TOPIC REPREZENTACIOKAT
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif model.custom_labels_ is not None and custom_labels:
        names = [model.custom_labels_[topic + model._outliers] for topic in unique_topics]
    else:
        names = [
            f"{topic}_" + "_".join([word for word, value in model.get_topic(topic)][:3])
            for topic in unique_topics
        ]
    
    
    if poet:
        df = df[df['poet'] == poet]
    
    if poem:
        
        df = df[df['title'] == poem]

    # Visualize
    fig = go.Figure()

 # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)

    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]


    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [
        None,
        None,
        selection.x.mean(),
        selection.y.mean(),
        'Other',
        'Other',
        "Other documents",

    ]
    
    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.doc if not hide_document_hover else None,
            hoverinfo="text",
            mode="markers+text",
            name="other",
            showlegend=False,
            marker=dict(color="#CFD8DC", size=5, opacity=0.5),
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [
                    None,
                    None,
                    selection.x.mean(),
                    selection.y.mean(),
                    name,
                ]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.doc if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode="markers+text",
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5),
                )
            )

    # Add grid in a 'plus' shape
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            "text": f"{'Vers Két Dimenzióban'}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        #yaxis_range = [-15,20],
        #xaxis_range = [-7,25],
        width=width,
        height=height,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

import dash
from dash import dcc, html, Input, Output

def run_app():
    app = dash.Dash(__name__)

    poets_list = viz_df['poet'].unique()
    poems_list = viz_df['title'].unique()

    # Layout
    app.layout = html.Div([
        html.H1("Poet Document Visualization"),
        dcc.Dropdown(
            id='poet_dropdown',
            options=[{'label': poet, 'value': poet} for poet in poets_list],
            value=None,
            clearable=True,
            style={'width': '100%'}
        ),
        dcc.Dropdown(
            id='poem_selector',
            options=[{'label': name, 'value': name} for name in poems_list],
            value=None,  # Default value
            searchable=True,  # Make the dropdown searchable
            placeholder="Select a name",
            clearable=True,  # Allow clearing the selection
            style={'width': '50%'}
        ),
        dcc.Graph(id='viz-graph')  # Placeholder for visualization
    ])

    # Callback to update poem selector options based on selected poet
    @app.callback(
        Output('poem_selector', 'options'),
        [Input('poet_dropdown', 'value')]
    )
    def update_poem_selector(poet_dropdown):
        if poet_dropdown is None:
            return [{'label': name, 'value': name} for name in poems_list]
        filtered_poems = viz_df[viz_df['poet'] == poet_dropdown]['title'].unique()
        return [{'label': name, 'value': name} for name in filtered_poems]

    # Callback to update poet dropdown options based on selected poem
    @app.callback(
        Output('poet_dropdown', 'value'),
        [Input('poem_selector', 'value')]
    )
    def update_poet_dropdown(poem_selector):
        if poem_selector is None:
            return None
        selected_poet = viz_df[viz_df['title'] == poem_selector]['poet'].iloc[0]
        return selected_poet

    # Callback for updating visualization based on selected poet and poem
    @app.callback(
        Output('viz-graph', 'figure'),
        [Input('poet_dropdown', 'value'),
        Input('poem_selector', 'value')]
    )
    def update_viz(poet_dropdown, poem_selector):
        print(poet_dropdown, poem_selector)
        fig = visualize_documents(
            viz_df,
            poet=poet_dropdown,
            poem=poem_selector,
            reduced_embeddings=umap2d_embeddings,
            width=1400, height=720,
            hide_annotations=True)
            #topics=list(range(16))
            # topics=list(range(15)))
        
        return fig

    # Run app
    if __name__ == '__main__':
        app.run_server(debug=False, port=9999)

run_app()


model.visualize_heatmap()




############## LDA #############
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel

def preprocess_data(documents):
    stop_words = hun_stopwords   
    # Tokenize and remove stopwords
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]  
    return texts

processed_texts = preprocess_data(docs)
id2word = corpora.Dictionary(processed_texts)
texts = processed_texts
corpus = [id2word.doc2bow(text) for text in texts]

def get_most_coherent_model(list_of_k:list):
    for i in list_of_k:
        num_topics = i
        lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10, alpha='auto', per_word_topics=True)

        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f'Coherence Score for k = {i}: ', coherence_lda)


#get_most_coherent_model(list_of_k=[18,19,22,25])
# 22 is most coherent

lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=22, random_state=42, passes=10, alpha='auto', per_word_topics=True)

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

pyLDAvis.enable_notebook()
p = gensimvis.prepare(lda_model, corpus=corpus, dictionary=id2word)
pyLDAvis.save_html(p, 'LDA.html')

###### EVAL OF BERTopic ########
viz_df2 = pd.DataFrame({
    'topic' : model.get_document_info(docs).iloc[:,1],
    'x' : umap2d_embeddings[:,0],
    'y' : umap2d_embeddings[:,1],
    'poem' :  insert_br_every_fifteen_words(docs2),
    'poet' : df['poet'],
    'title' : df['Cim'],
    'doc' : docs
})
documents_per_topic = viz_df2.groupby(['topic'], as_index=False).agg({'doc': ' '.join})

cleaned_docs =model._preprocess_text(documents_per_topic.doc.values)

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
