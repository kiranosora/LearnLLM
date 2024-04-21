# TF-IDF: term frequency - inverse document frequency  词频-逆向文档频率
# 文档首先去掉'a'等词
# TF = 每个文档中每个词出现次数
# DF = 每个词出现的文档个数 + 1
# IDF = ln（(文档个数 +1）/DF） + 1
# TF-IDF = TF * IDF / l2_norm
# l2_norm为每个文档的 TF * IDF 向量平方和的平方根，为归一化因子。保证每个文档的TF-IDF向量的欧几里得长度为1。

import math
class MyTFIDF:
    ignore_words = ['a']
    def __init__(self):
        pass

    #统计词频
    def count_features(self, raw_docs: [[str]]):
        counts = []
        vocabulary = dict()
        for doc in raw_docs:
            count_f = dict()
            for word in doc:
                if word in count_f:
                    count_f[word] += count_f[word]
                else:
                    count_f[word] = 1
                if word not in vocabulary:
                    vocabulary[word] = len(vocabulary)
            counts.append(count_f)
        word_keys = sorted(vocabulary.keys())
        vocabulary = dict((word_keys[idx], idx) for idx in range(len(word_keys)))
        print(f"counts: {counts}")
        print(f"vocabulay: type:{type(vocabulary)},  {vocabulary}")
        res_counts = []
        for idx in range(len(counts)):
            res_count = dict()
            for word in counts[idx].keys():
                res_count[vocabulary[word]] = counts[idx][word]
            res_counts.append(res_count)
        return vocabulary, res_counts


    def counter(self, documents:[str]):
        raw_docs = []
        for doc in documents:
            a_doc = doc.split()
            for word in MyTFIDF.ignore_words:
                if word in a_doc:
                    a_doc.remove(word)
            raw_docs.append(a_doc)
        return self.count_features(raw_docs=raw_docs)

    def fit_transform(self, documents: [str]):
        raw_docs = []
        for doc in documents:
            a_doc = doc.split()
            for word in MyTFIDF.ignore_words:
                if word in a_doc:
                    a_doc.remove(word)
            raw_docs.append(a_doc)

        documents = raw_docs

        num_doc = len(documents)
        tf = []
        df = dict()
        for idx in range(num_doc):
            doc_dict = dict()
            words = documents[idx]
            for word in words:
                if word in doc_dict:
                    doc_dict[word] += 1
                else:
                    doc_dict[word] = 1
            for word in doc_dict.keys():
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 2
                # if word in doc_dict.keys():
                #     doc_dict[word] = doc_dict[word]/len(words)
            tf.append(doc_dict)
            print(f"doc_dict: {doc_dict}")

        print(f"df: {df}")

        idf = dict()
        for word in df.keys():
            idf[word] = math.log((num_doc + 1)/(df[word])) + 1
            print(f"idf[{word}] =  {idf[word]}, num_doc = {num_doc}, df[{word}] = {df[word]}")

        tdidfs = []
        for idx in range(num_doc):
            tfidf = dict()
            l2_norm = 0
            for word in df.keys():
                if word in tf[idx]:
                    tfidf[word] = tf[idx][word] * idf[word]
                    l2_norm += tfidf[word] * tfidf[word]
                else:
                    tfidf[word] = 0
                if word == 'a':
                    print(f"tf[{idx}]: {tf[idx]}  tfidf:{tfidf}  idf:{idf}")

            #perform l2 norm
            l2_norm = math.sqrt(l2_norm)
            for word in df.keys():
                tfidf[word] = tfidf[word] / l2_norm

            tdidfs.append(tfidf)
        return tdidfs


if __name__ == '__main__':
    documents = ["the fat cat sat on the mat",
                 "the big cat slept",
                 "the dog chased a cat"]
    tf_idf = MyTFIDF()
    results = tf_idf.fit_transform(documents)
    for res in results:
        res = {k:v for k, v in sorted(res.items(), reverse=False)}
        print(f"res.keys(): {len(res.keys())},  res: {res}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import pandas as pd
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(documents)
    vocabulary, counters = tf_idf.counter(documents)
    print(f"vocabulary: {vocabulary}")
    print(f"counters: {counters}")
    print(f"The vocabulary size is {len(vectorizer.vocabulary_.keys())}")
    print(f"The document - term matrix shape is {corpus_tfidf.shape}")
    df = pd.DataFrame(np.round(corpus_tfidf.toarray(), 2))
    print(f"corpus_tfidf: {corpus_tfidf}")




