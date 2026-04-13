import json
import re
import joblib
from wordcloud import wordcloud

import cluster_kmeans


def get_all_docs():
    return []

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


def training_model(get_docs_from_local):
    print('start to train model')
    data = get_all_docs.query_data(get_docs_from_local)
    print('train model - read data complete')
    cluster_kmeans.KmeansClusterClass(500, data)


def predict(x, cluster_n, raw_message):
    loaded_kmeans = joblib.load("clustering/kmeans_model_" + str(cluster_n) + ".pkl")
    if loaded_kmeans is None:
        return ""
    predict_result = loaded_kmeans.predict(x)
    tfidf_vectorizer = loaded_kmeans.named_steps['tfidf']
    vectors = tfidf_vectorizer.transform(x)
    vector_array = vectors.toarray()
    print('tfidf word count:', len(tfidf_vectorizer.get_feature_names_out()))

    with open("clustering/key_words_" + str(cluster_n), "r+") as outfile:
        key_words_file = outfile.read()
        key_words_str = json.loads(key_words_file)[str(predict_result[0])]
        key_words_list = key_words_str.split(',')
        # error_info = key_words_list[predict_result[0]].split(',')
        for word in key_words_list:
            if re.search(str(word).lower(), raw_message.lower(), re.IGNORECASE):
                return key_words_str
    return ""

def split_rca_str(rca_str):
    rca_str = str(rca_str).lower()
    #split by . , and space
    seps = [",", ".", "$", "!", "?", "&", ";", ":", "[", "]", "(", ")"]
    for sep in seps:
        rca_str = str(rca_str).replace(sep, " ")
    return rca_str.split()


def predict_db_scan(x, eps, min_samples, raw_message):
    loaded_kmeans = joblib.load('clustering/DBScan_model_' + str(eps) + '_' + str(min_samples) + '.pkl')
    if loaded_kmeans is None:
        return ""
    predict_result = loaded_kmeans.predict(x)
    with open("clustering/DBScan_key_words_" + str(eps) + '_' + str(min_samples), "r+") as outfile:
        key_words_file = outfile.read()
        key_words_str = json.loads(key_words_file)[str(predict_result[0])]
        key_words_list = key_words_str.split(',')
        # error_info = key_words_list[predict_result[0]].split(',')
        for word in key_words_list:
            if word in raw_message[0]:
                return key_words_str
        return ""


def predict_batch(x, cluster_n):
    loaded_kmeans = joblib.load("clustering/kmeans_model_" + str(cluster_n) + ".pkl")
    if loaded_kmeans is None:
        return {}
    predict_result = loaded_kmeans.predict(x)
    result = []
    with open("clustering/key_words_" + str(cluster_n), "r+") as outfile:
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
    #training_model(False)

    r = predict(['java lang OutOfMemoryError'], 500, 'java.lang.OutOfMemoryError')
    print(r)

    print('')
