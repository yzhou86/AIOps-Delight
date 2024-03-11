import json
import time

import joblib
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import sentence_vec
from cluster_category_util import cluster_category


def send_kafka(messages):
    pass

class KmeansClusterClass:
    def __init__(self, model_n, data):
        self.model_n = model_n
        self.all_docs = data

        # self.training(data, first_model_n)
        self.training(data, model_n)

    def training(self, data, n_cluster):
        print('train model - start tfidf and kmeans training')
        new_docs = []
        new_category = []
        for item in data:
            if item['doc'] != "":
                for i in range(0, item['count(*)'] + 1):
                    new_docs.append(item['doc'])
                    new_category.append(item['category'])
        tf_idf_vec, tf_idf = sentence_vec.tf_idf_vec(new_docs)
        print('train model - sentence vector done')

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1, 1))),
            ('kmeans', KMeans(n_clusters=n_cluster, random_state=1000))
        ])
        df_array = []
        cluster_labels = pipeline.fit_predict(new_docs)
        print('train model - pipeline done')
        docs = {"doc": new_docs, "cluster": cluster_labels}
        df = DataFrame(docs)
        keyword_json = {}
        original_category = {}
        for i in range(0, n_cluster):
            df_array.append(df[df["cluster"] == i])
        all_ratio_message = []
        timestamp = int(time.time() * 1000)
        count_array = []
        for cluster_idx in list(set(cluster_labels)):
            cluster_text_indices = np.where(cluster_labels == cluster_idx)[0]
            count_array.append(len(cluster_text_indices))
            cluster_keywords = []
            original_doc_category = []
            for idx in cluster_text_indices:
                cluster_keywords.extend(tf_idf_vec[idx].nonzero()[1])
                original_doc_category.append(new_category[idx])
            cluster_keywords = list(set(cluster_keywords))
            cluster_keywords = sorted(cluster_keywords, key=lambda x: tf_idf_vec[cluster_text_indices[0], x],
                                      reverse=True)
            if len(cluster_keywords) == 0:
                print()
            top_keywords = [tf_idf.get_feature_names_out()[idx] for idx in cluster_keywords[:3]]
            print(f"Cluster {cluster_idx + 1}  center word: {', '.join(top_keywords)}")
            keyword_json[str(cluster_idx)] = ','.join(top_keywords)

            # send kafka

            tag = {}
            key_word_str = ','.join(top_keywords)
            category = key_word_str
            tag["category"] = category

            ratio_message = {"featureName": "test_unified_monitor", "metric_type": "test_aiops_rca_top",
                             "number_key": "count", "number_value": len(cluster_text_indices), "tags": tag,
                             "timestamp": timestamp}

            all_ratio_message.append(ratio_message)

            # action = get_action_item(category)
            action = cluster_category(category)
            original_category[str(cluster_idx)] = {'key_word': category,
                                                   'action': action,
                                                   'original_category': list(set(original_doc_category))}

        with open("clustering/key_words_" + str(n_cluster), "w") as outfile:
            outfile.write(json.dumps(keyword_json))
        with open("clustering/original_category_" + str(n_cluster), "w") as outfile:
            outfile.write(json.dumps(original_category))
        if n_cluster == 50:
            send_kafka(all_ratio_message)
        joblib.dump(pipeline, 'clustering/kmeans_model_' + str(n_cluster) + '.pkl')

    def training2(self, data, n_cluster):
        print('train model - start tfidf and kmeans training')
        new_docs = []
        new_category = []
        for item in data:
            if item['doc'] != "":
                for i in range(0, item['count(*)'] + 1):
                    new_docs.append(item['doc'])
                    new_category.append(item['category'])
        tf_idf_vec, tf_idf = sentence_vec.tf_idf_vec(new_docs)
        print('train model - sentence vector done')

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1, 1))),
            ('kmeans', KMeans(n_clusters=n_cluster, n_init=20, max_iter=600))
        ])
        df_array = []
        cluster_labels = pipeline.fit_predict(new_docs)
        print('train model - pipeline done')
        docs = {"doc": new_docs, "cluster": cluster_labels}
        df = DataFrame(docs)
        keyword_json = {}
        original_category = {}
        for i in range(0, n_cluster):
            df_array.append(df[df["cluster"] == i])
        all_ratio_message = []
        timestamp = int(time.time() * 1000)
        count_array = []
        for cluster_idx in list(set(cluster_labels)):
            cluster_text_indices = np.where(cluster_labels == cluster_idx)[0]
            count_array.append(len(cluster_text_indices))
            cluster_keywords = []
            original_doc_category = []
            for idx in cluster_text_indices:
                cluster_keywords.extend(tf_idf_vec[idx].nonzero()[1])
                original_doc_category.append(new_category[idx])

            first_n_keys = get_sort_words(cluster_text_indices, tf_idf_vec, 3)
            top_keywords = [tf_idf.get_feature_names_out()[idx] for idx in first_n_keys]
            print(f"Cluster {cluster_idx + 1}  center word: {', '.join(top_keywords)}")
            keyword_json[str(cluster_idx)] = ','.join(top_keywords)

            # send kafka

            tag = {}
            key_word_str = ','.join(top_keywords)
            category = key_word_str
            tag["category"] = category

            ratio_message = {"featureName": "test_unified_monitor", "metric_type": "test_aiops_rca_top",
                             "number_key": "count", "number_value": len(cluster_text_indices), "tags": tag,
                             "timestamp": timestamp}

            all_ratio_message.append(ratio_message)

            # action = get_action_item(category)
            action = cluster_category(category)
            original_category[str(cluster_idx)] = {'key_word': category,
                                                   'action': action,
                                                   'original_category': list(set(original_doc_category))}

        with open("clustering/key_words_" + str(n_cluster), "w") as outfile:
            outfile.write(json.dumps(keyword_json))
        with open("clustering/original_category_" + str(n_cluster), "w") as outfile:
            outfile.write(json.dumps(original_category))
        if n_cluster == 50:
            send_kafka(all_ratio_message)
        joblib.dump(pipeline, 'clustering/kmeans_model_' + str(n_cluster) + '.pkl')

    def debug_training(self, all_docs, n_cluster):

        tf_idf_vec, tf_idf = sentence_vec.tf_idf_vec(all_docs)
        # num_clusters = 10/13
        km = KMeans(random_state=10000, n_clusters=n_cluster)
        # %time km.fit(tfidf_matrix)
        # k = np.arange(1,20)
        # jarr = []
        # for i in k:
        #     model = KMeans(n_clusters=i)
        #     model.fit(word_vector)
        #     jarr.append(model.inertia_)
        #     plt.annotate(str(i), (i, model.inertia_))
        # plt.plot(k,jarr)
        # plt.show(), 'o-')

        km.fit_transform(tf_idf_vec)
        lables = km.labels_.tolist()
        # kmeans_plot(km)
        df_array = []
        docs = {"doc": all_docs, "cluster": lables}
        df = DataFrame(docs)
        for i in range(0, n_cluster):
            df_array.append(df[df["cluster"] == i])
        cluster_labels = km.labels_
        keyword_list = []
        keyword_json = {}
        cluster_centers_indices = np.argsort(np.linalg.norm(km.cluster_centers_, axis=1))

        for cluster_idx in cluster_centers_indices:
            cluster_text_indices = np.where(cluster_labels == cluster_idx)[0]
            cluster_keywords = []
            for idx in cluster_text_indices:
                cluster_keywords.extend(tf_idf_vec[idx].nonzero()[1])
            # first_n_keys = get_sort_words(cluster_text_indices,tf_idf_vec, 3)

            cluster_keywords = list(set(cluster_keywords))
            cluster_keywords = sorted(cluster_keywords, key=lambda x: tf_idf_vec[cluster_text_indices[0], x],
                                      reverse=True)
            # top_keywords = [tf_idf.get_feature_names_out()[idx] for idx in first_n_keys]

            top_keywords = [tf_idf.get_feature_names_out()[idx] for idx in cluster_keywords[:5]]
            print(f"Cluster {cluster_idx + 1}  center word: {', '.join(top_keywords)}")
            keyword_json[str(cluster_idx)] = ','.join(top_keywords)
        with open("clustering/key_words_" + str(n_cluster), "w") as outfile:
            outfile.write(json.dumps(keyword_json))
        print()


def get_category_by_keyword(model_type, rca_message):
    rca_message = str(rca_message).lower()
    if model_type == 50:
        with open("clustering/50_category.json", 'r') as outfile:
            cluster_keyword_10 = json.loads(outfile.read())
            for key in cluster_keyword_10.keys():
                is_cluster = True
                keywords = cluster_keyword_10.get(key).split(",")
                for keyword in keywords:
                    if keyword not in rca_message:
                        is_cluster = False
                if is_cluster:
                    return key
            return rca_message
    if model_type == 100:
        with open("clustering/100_category.json", 'r') as outfile:
            cluster_keyword_50 = json.loads(outfile.read())
            for key in cluster_keyword_50.keys():
                is_cluster = True
                keywords = cluster_keyword_50.get(key).split(",")
                for keyword in keywords:
                    if keyword not in rca_message:
                        is_cluster = False
                if is_cluster:
                    return key
            return rca_message


def split_rca_str(rca_str):
    rca_str = str(rca_str).lower()
    # split by . , and space
    seps = [",", ".", "$", "!", "?", "&", ";", ":", "[", "]", "(", ")"]
    for sep in seps:
        rca_str = str(rca_str).replace(sep, " ")
    return rca_str.split()


def get_action_item(rca_str):
    action = ""
    # if rca_str:
    #     keywords = split_rca_str(rca_str)
    #     for item in action_mapping:
    #         action_kw = item.get('keywords')
    #         if set(keywords) >= set(action_kw):
    #             action = item.get('action')
    return action


def get_sort_words(cluster_text_indices, tf_idf_vec, first_n=3):
    result_map = {}
    tf_idf_vec_current = tf_idf_vec[cluster_text_indices]
    all_col_sum = np.sum(tf_idf_vec_current, axis=0).T
    for i in range(0, all_col_sum.shape[0] - 1):
        result_map[i] = float(all_col_sum[i])

    sorted_map = sorted(result_map.items(), key=lambda x: x[1], reverse=True)
    list_items = sorted_map[0:first_n]
    list_keys = [x[0] for x in list_items]
    return list_keys
