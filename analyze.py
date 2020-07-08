#   -*- coding: utf-8 -*-
#
#   analyze.py
#   
#   Developed by Tianyi Liu on 2020-05-27 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import os
import time
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN, AffinityPropagation, OPTICS
import phenograph


from cfg import *
from utils import check_directory, cache_operation


def run_dr(data, dr_type, epoch=None, args=None, dim=2, cache=True):
    """
    Running TSNE/UMAP/PCA DR
    :param data: Original data
    :param dr_type: DR type TSNE/UMAP/PCA
    :param epoch: (Optional) -> DR for NN embedding
    :param args: (Optional) -> filtered data
    :param dim: DR dimension
    :param cache: Writing DR cache
    :return: embedding
    """
    def _dr(_data, _dr_type, _dim):
        print(">>> Running {} embedding".format(_dr_type.upper()))
        if np.array(_data).shape[1] == _dim:
            return _data
        tic = time.time()
        if _dr_type.upper() == "TSNE":
            _embedding = TSNE(n_components=2, n_jobs=TSNE_JOBS).fit_transform(_data)
        elif _dr_type.upper() == "UMAP":
            _embedding = umap.UMAP(n_neighbors=5, n_components=_dim).fit_transform(_data)
        elif _dr_type.upper() == "PCA":
            _embedding = PCA(n_components=_dim).fit_transform(data)
        else:
            raise Exception("!!! Invalid dr_type provided.")
        toc = time.time()
        print("    {} took {:.2f} s".format(_dr_type.upper(), toc - tic))
        return _embedding

    if epoch is None:
        file_name = dr_type.lower() + "_data_cache_"
        # Check cache
        if args is None and os.path.isfile(os.path.join(CACHE_DIR, file_name + "{}d.pkl".format(dim))):
            embedding = cache_operation(
                os.path.join(CACHE_DIR, file_name + "{}d.pkl".format(dim)), "read", text=" " + dr_type.upper())
        elif args is not None and os.path.isfile(os.path.join(CACHE_DIR, file_name + "{}m_{}s_{}d.pkl".format(
                args.mean_filter, args.sd_filter, dim))):
            embedding = cache_operation(os.path.join(CACHE_DIR, file_name + "{}m_{}s_{}d.pkl".format(
                    args.mean_filter, args.sd_filter, dim)), "read", text=" " + dr_type.upper())
        else:
            embedding = _dr(data, dr_type, dim)
            if cache:
                check_directory(CACHE_DIR)
                write_name = file_name + "{}d.pkl".format(
                    dim) if args is None else file_name + "{}m_{}s_{}d.pkl".format(
                    args.mean_filter, args.sd_filter, dim)
                cache_operation(os.path.join(CACHE_DIR, write_name), "write", embedding, text=" " + dr_type.upper())
    else:
        embedding = _dr(data, dr_type, dim)
    return embedding


def plot_embedding(embedding, assignment=None, label=None, batch_correction=None,
                   title=None, epoch=None, show=False, dr_type="TSNE"):
    """
    Visualize embedding
    :param embedding: embedding
    :param assignment: (Optional) clustering assignment
    :param label: (Optional) ground truth assignment
    :param title: (Optional) plot title
    :param epoch: (Optional) visualize epoch
    :param show: show plots
    :param dr_type: embedding type
    :return: none
    """

    def _2d_plot(embedding, data, title, anno=False):
        unique_data = np.unique(data)
        cmap = plt.get_cmap("Spectral")
        colors = cmap(np.linspace(0, 1.0, len(unique_data)))
        for item, color in zip(unique_data, colors):
            plt.scatter(embedding[data == item, 0], embedding[data == item, 1], s=1, label=item, c=[color])
        if anno:
            for item, txt in enumerate(data):
                if item % 50 == 0:
                    plt.annotate(txt, (embedding[item, 0], embedding[item, 1]))
        plt.title(title)
        plt.legend(loc="upper right")

    embedding = np.array(embedding)
    if embedding.shape[1] != 2:
        print("!!! 2D embedding expected for visualization while {}D provided".format(embedding.shape[1]))

    check_directory(VISUL_DIR)
    if assignment is None and label is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=1)
        plt.title(dr_type + " Embedding of {}".format(title))
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(VISUL_DIR, dr_type.lower() + "_{}.pdf".format(title)), dpi=400)

    elif assignment is not None and label is None:
        _2d_plot(embedding, assignment,
                 "Cluster Assignment Epoch={}".format(epoch + 1), False)
        plt.savefig(os.path.join(VISUL_DIR, dr_type.lower() + "_cls_epoch_{}.pdf".format(epoch + 1)), dpi=400)

    elif assignment is None and label is not None:
        if epoch is None:
            title = "Labels"
            filename = os.path.join(VISUL_DIR, dr_type.lower() + "_label.pdf")
        else:
            title = "DR with Labels Epoch={}".format(epoch + 1)
            filename = os.path.join(VISUL_DIR, dr_type.lower() + "_label_epoch_{}.pdf".format(epoch + 1))
        _2d_plot(embedding, label, title, False)
        plt.savefig(filename, dpi=400)

    elif assignment is not None and label is not None and batch_correction is None:
        plt.figure(figsize=(15, 8))
        plt.subplot(121)
        _2d_plot(embedding, assignment, "Cluster Assignment Epoch={}".format(epoch + 1), True)
        plt.subplot(122)
        _2d_plot(embedding, label, "Labels", True)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUL_DIR, dr_type.lower() + "_cls_label_epoch{}.pdf".format(epoch + 1)), dpi=400)

    elif assignment is not None and label is not None and batch_correction is not None:
        plt.figure(figsize=(21, 8))
        plt.subplot(131)
        _2d_plot(embedding, assignment, "Cluster Assignment Epoch={}".format(epoch + 1), True)
        plt.subplot(132)
        _2d_plot(embedding, batch_correction, "Batch Correction Epoch={}".format(epoch + 1), False)
        plt.subplot(133)
        _2d_plot(embedding, label, "Labels", True)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUL_DIR, dr_type.lower() + "_cls_bc_label_epoch{}.pdf".format(epoch + 1)), dpi=400)

    if show:
        plt.show()
    plt.clf()


def run_k_means(data, n_clusters):
    print(">>> Running K-Means with {} clusters".format(n_clusters))
    clf = KMeans(n_clusters=n_clusters)
    tic = time.time()
    clf.fit(data)
    toc = time.time()
    print("    K-Means took {:.2f} s".format(toc - tic))
    return clf.labels_


def run_spectral_clustering(data, n_clusters):
    print(">>> Running Spectral Clustering with {} clusters".format(n_clusters))
    clf = SpectralClustering(n_clusters=n_clusters)
    tic = time.time()
    clf.fit(data)
    toc = time.time()
    print("    Spectral Clustering took {:.2f} s".format(toc - tic))
    return clf.labels_


def run_dbscan(data, eps=3):
    print(">>> Running DBSCAN")
    clf = DBSCAN(eps=eps)
    tic = time.time()
    clf.fit(data)
    toc = time.time()
    print("    DBSCAN found {} clusters".format(len(np.unique(clf.labels_))))
    print("    DBSCAN took {:.2f} s".format(toc - tic))
    return clf.labels_


def run_ap(data):
    print(">>> Running Affinity Propagation")
    clf = AffinityPropagation()
    tic = time.time()
    clf.fit(data)
    toc = time.time()
    print("    Affinity Propagation found {} clusters".format(len(np.unique(clf.labels_))))
    print("    Affinity Propagation took {:.2f} s".format(toc - tic))
    return clf.labels_


def run_optics(data):
    print(">>> Running OPTICS")
    clf = OPTICS(min_samples=25, xi=.05, min_cluster_size=.02)
    tic = time.time()
    clf.fit(data)
    toc = time.time()
    print("    OPTICS found {} clusters".format(len(np.unique(clf.labels_))))
    print("    OPTICS took {:.2f} s".format(toc - tic))
    return clf.labels_


def run_phenograph(data):
    print(">>> Running PhenoGraph")
    tic = time.time()
    communities, _, _ = phenograph.cluster(data)
    toc = time.time()
    print("    PhenoGraph found {} clusters".format(len(np.unique(communities))))
    print("    PhenoGraph took {:.2f} s".format(toc - tic))
    return communities
