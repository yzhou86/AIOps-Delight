from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf_vec(all_docs):
    tv = TfidfVectorizer(use_idf=True, ngram_range=(1, 1))
    tv_fit = tv.fit_transform(all_docs)
    print("after fit, all vocabulary is as follows:")
    features = tv.get_feature_names_out()
    print(tv.get_feature_names_out())
    print("after fit, The vectorization of the training data is expressed as:")
    print(tv_fit)
    return tv_fit, tv



