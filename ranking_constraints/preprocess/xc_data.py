import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from xclib.data import data_utils

from config import dataConfig as dc

RELEVANT_THRES = 11


if __name__ == "__main__":
    # Save the Top K documents with relevant queries to doc_id.txt
    dc.init_config("xc")
    DATA_DIR, EXP_DIR, K = dc.DATA_DIR, dc.EXP_DIR, dc.K

    train_path = DATA_DIR / "train.txt"
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        train_path
    )

    num_samples_per_label = np.asarray(labels.sum(axis=0)).reshape(-1)
    doc_ids = list(np.argsort(num_samples_per_label)[-K:])
    doc_ids_path = EXP_DIR / "doc_ids.txt"
    np.savetxt(doc_ids_path, doc_ids, fmt="%s")

    # Save the relevant documents for each query to query2doc.pkl
    train_id_path = DATA_DIR / "train_raw_ids.txt"
    train_ids = np.loadtxt(train_id_path, dtype=int)
    query2doc = defaultdict(list)
    for train_id in tqdm(train_ids):
        row = features[train_id]
        _, cols = row.nonzero()
        if len(cols) == 0:
            continue
        for doc_id in doc_ids:
            if labels[train_id, doc_id]:
                query2doc[train_id].append(doc_id)

    # Pick the queries with >= RELEVANT_THRES relevant docments and save the list to query_id.txt
    max_docs = max([len(query2doc[query_id]) for query_id in query2doc])
    num_docs_hist = np.zeros(max_docs + 1, dtype=int)
    for query_id in query2doc:
        num_docs_hist[len(query2doc[query_id])] += 1
    for i in range(max_docs + 1):
        print(f"#Relevent documents >= {i}: {sum(num_docs_hist[i:])} queries.")
    query_ids = [
        query_id for query_id in query2doc if len(query2doc[query_id]) >= RELEVANT_THRES
    ]
    query_ids_path = EXP_DIR / "query_ids.txt"
    np.savetxt(query_ids_path, query_ids, fmt="%s")

    # Save the (query_id, doc_id) pairs to relevance.pkl
    relevance = set()
    for query_id in query_ids:
        for doc_id in doc_ids:
            if labels[query_id, doc_id]:
                relevance.add((query_id, doc_id))
    pickle.dump(relevance, open(EXP_DIR / "relevance.pkl", "wb"))
