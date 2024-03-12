# SPDX-FileCopyrightText: 2024-present Henry Watkins <h.watkins@ucl.ac.uk>
#
# SPDX-License-Identifier: MIT
from numba import jit
import numpy as np



@jit(nopython=True, parallel=True)
def numba_cluster_feat_mccs(cluster_id, feature_vecs, cluster_vec):
    """numba-accelerated matthews correlation coefficient calculation
    for finding base features most heavily correlated with cluster ids"""
    mccs = []
    for feature_idx in range(feature_vecs.shape[1]):
        in_cluster = cluster_vec == cluster_id
        has_feature = feature_vecs[:, feature_idx] > 0.0
        tp = np.logical_and(in_cluster, has_feature).sum()
        fp = np.logical_and(np.logical_not(in_cluster), has_feature).sum()
        tn = np.logical_and(
            np.logical_not(in_cluster), np.logical_not(has_feature)
        ).sum()
        fn = np.logical_and(in_cluster, np.logical_not(has_feature)).sum()
        num = tp * tn - fp * fn
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        MCC = num / denom
        mccs.append(MCC)
    mccs = np.array(mccs)
    return mccs


def one_hot_encode(X):
    """one-hot encode a list of lists of data"""
    classes = sorted(list({item for sublist in X for item in sublist}))
    encoded = np.zeros((len(X), len(classes)), dtype=int)
    indices = [(i, classes.index(item)) for i, row in enumerate(X) for item in row]
    rows, cols = zip(*indices)
    encoded[rows, cols] = 1
    return encoded, classes


def cfeatures(group_ids, data ,top_k=5, show_values=False):
    """Find the top k features most heavily correlated with each group id"""
    vecs, vocab = one_hot_encode(data)
    cluster_labels = np.array(group_ids)
    cluster_mccs = {}
    for cl in np.unique(cluster_labels):
        mccs = numba_cluster_feat_mccs(cl, vecs, cluster_labels)
        cluster_mccs[cl] = mccs
    reversed_vocab = dict(enumerate(vocab))
    top_cluster_vars = {}
    for cl in cluster_mccs.keys():
        top_idxs = np.argsort(cluster_mccs[cl])[-top_k:]
        if show_values:
            top_vars = [(reversed_vocab[i], cluster_mccs[cl][i]) for i in top_idxs]
        else:
            top_vars = [reversed_vocab[i] for i in top_idxs]
        #top_vars = [reversed_vocab[i] for i in top_idxs]
        top_cluster_vars[cl] = top_vars
    return top_cluster_vars
