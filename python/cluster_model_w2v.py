import json
import re

import gensim.models
import joblib
import pandas as pd
from wordcloud import wordcloud

from clustering import get_all_docs
import pickle
from gensim.models import word2vec
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import defaultdict


_WORD2VEC_MODEL_PATH = 'clustering/word2vec.model'
_WORD2VEC_KMEANS_PATH = 'clustering/word2vec_kmeans.pkl'
_WORD2VEC_KEYWORDS_PATH = 'clustering/word2vec_keywords.pkl'


def generate_word_cloud():
    all_docs = get_all_docs.get_all_docs(True)
    wc = wordcloud.WordCloud(
        width=1000,
        height=700,
        background_color='white',
        collocations=False)

    text = " ".join(all_docs)

    wc.generate(text)
    wc.to_file("clustering/wordcloud.png")


def load_word2vec():
    return gensim.models.Word2Vec.load(_WORD2VEC_MODEL_PATH)


def load_kmeans():
    return joblib.load(_WORD2VEC_KMEANS_PATH)


def training_model(get_docs_from_local):
    print('start to train model word2vec')
    data = get_all_docs.query_data(get_docs_from_local)
    print('train model word2vec - read data complete')
    print('train model - start tfidf and kmeans training')
    new_docs = []
    new_category = []
    for item in data:
        if item['doc'] != "":
            new_docs.append(item['doc'].split(' '))
            new_category.append(item['category'])

    model = word2vec.Word2Vec(new_docs, min_count=1, workers=10)
    model.save(_WORD2VEC_MODEL_PATH)
    print('train model - sentence vector done')

    features = []

    for item in data:
        if item['doc'] != "":
            current_vec = get_vec(item['doc'], model)
            features.append(current_vec)


    vectorized_docs = features
    print('vectorized_docs:', len(vectorized_docs))

    km_model, cluster_labels = mbkmeans_clusters(vectorized_docs, 300, 500, True)
    with open(_WORD2VEC_KMEANS_PATH, 'wb') as km_f:
        pickle.dump(km_model, km_f)
    print('cluster done')

    cluster_category = []
    for idx, category in enumerate(new_category):
        category_with_cluster = {
            'category': category,
            'cluster': list(cluster_labels)[idx]
        }
        cluster_category.append(category_with_cluster)


    print("Most representative terms per cluster (based on centroids):")
    keywords_dict = {}
    for i in range(300):
        tokens_per_cluster = ""
        most_representative = model.wv.most_similar(positive=[km_model.cluster_centers_[i]], topn=5)
        for t in most_representative:
            tokens_per_cluster += f"{t[0]},"
        tokens_per_cluster = tokens_per_cluster[:-1]
        print(f"Cluster {i}: {tokens_per_cluster}")
        keywords_dict[i] = tokens_per_cluster
    with open(_WORD2VEC_KEYWORDS_PATH, "w") as outfile:
        outfile.write(json.dumps(keywords_dict))


    temp = defaultdict(list)
    for item in cluster_category:
        key = item.get('cluster')
        val = item.get('category')
        temp[key].append(val)
    groups = dict((key, tuple(val)) for key, val in temp.items())
    print('group done')


def get_vec(doc, word2vec_model):
    tokens = doc.split(' ')
    zero_vector = np.zeros(word2vec_model.vector_size)
    vectors = []
    for token in tokens:
        if token in word2vec_model.wv:
            try:
                vectors.append(word2vec_model.wv[token])
            except KeyError:
                continue
    if vectors:
        vectors = np.asarray(vectors)
        avg_vec = vectors.mean(axis=0)
        return avg_vec
    else:
        return zero_vector



def mbkmeans_clusters(
        X,
        k,
        mb,
        print_silhouette_values,
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            if len(cluster_silhouette_values) > 0:
                silhouette_values.append(
                    (
                        i,
                        cluster_silhouette_values.shape[0],
                        cluster_silhouette_values.mean(),
                        cluster_silhouette_values.min(),
                        cluster_silhouette_values.max(),
                    )
                )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )

    return km, km.labels_


def predict(doc, cluster_n, raw_message):
    loaded_kmeans = load_kmeans()
    loaded_word2vec = load_word2vec()
    if loaded_kmeans is None:
        return ""

    vec = get_vec(doc[0], loaded_word2vec)
    predict_result = loaded_kmeans.predict([vec])
    with open(_WORD2VEC_KEYWORDS_PATH, "r+") as outfile:
        key_words_file = outfile.read()
        key_words_str = json.loads(key_words_file)[str(predict_result[0])]
        key_words_list = key_words_str.split(',')
        for word in key_words_list:
            if re.search(str(word).lower(), raw_message.lower(), re.IGNORECASE):
                return key_words_str
    return ""


def split_rca_str(rca_str):
    rca_str = str(rca_str).lower()
    # split by . , and space
    seps = [",", ".", "$", "!", "?", "&", ";", ":", "[", "]", "(", ")"]
    for sep in seps:
        rca_str = str(rca_str).replace(sep, " ")
    return rca_str.split()


def predict_batch(x, cluster_n):
    loaded_kmeans = load_kmeans()
    if loaded_kmeans is None:
        return {}
    loaded_word2vec = load_word2vec()
    vectors = []
    for item in x:
        vec = get_vec(item, loaded_word2vec)
        vectors.append(vec)
    predict_result = loaded_kmeans.predict(vectors)
    result = []
    with open(_WORD2VEC_KEYWORDS_PATH, "r+") as outfile:
        key_words_file = outfile.read()
        key_words_json = json.loads(key_words_file)
        for i in range(0, len(predict_result)):
            raw_message = x[i]
            key_words_str = key_words_json[str(predict_result[i])]
            key_words_list = key_words_str.split(',')
            for word in key_words_list:
                if re.search(str(word).lower(), raw_message.lower(), re.IGNORECASE):
                    result.append(key_words_str)
                    break
            if len(result) < i + 1:
                result.append("")
    return result


if __name__ == '__main__':
    # training_model(True)
    # training_model(False)

    r = predict(['java lang OutOfMemoryError'], "", 'java.lang.OutOfMemoryError')
    print(r)

    print('')
