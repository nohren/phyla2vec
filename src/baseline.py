# knn baseline model, ingest 3/4 of test set as training data, predict using the remaining 1/4 of samples (this exact process is repeated with transformer encoder model for comparison)

import unifrac 
import numpy as np
import BiomDataset as bd

TREE_FP = "tree.nwk"  # your (possibly filtered) GG2 tree

def unifrac_knn_predict_one(test_vec, X_train, y_train, otu_ids, tree_fp, k=5):
    """
    test_vec:  shape (n_features,)
    X_train:   shape (n_train, n_features)
    y_train:   shape (n_train,)
    otu_ids:   list/array of ASV IDs, length n_features
    """
    dists = []
    for train_vec in X_train:
        d = unifrac.unweighted_dense_pair(otu_ids, test_vec, train_vec, tree_fp)
        dists.append(d)

    dists = np.array(dists)
    nn_idx = np.argsort(dists)[:k]
    nn_labels = y_train[nn_idx]

    # majority vote
    vals, counts = np.unique(nn_labels, return_counts=True)
    return vals[np.argmax(counts)]

def unifrac_knn_predict_batch(X_test, X_train, y_train, otu_ids, tree_fp, k=5):
    preds = []
    for i in range(X_test.shape[0]):
        y_hat = unifrac_knn_predict_one(X_test[i], X_train, y_train, otu_ids, tree_fp, k=k)
        preds.append(y_hat)
    return np.array(preds)
