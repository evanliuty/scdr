#   -*- coding: utf-8 -*-
#
#   utils.py
#   
#   Developed by Tianyi Liu on 2020-05-26 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import os
import anndata
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle as pkl
from anndata import AnnData


from cfg import *
from preproc import label_str_to_int


def load_data(args):
    def _load_data(args):
        # Read cache
        if args.read_cache is not None:
            if os.path.isfile(args.read_cache):
                adata = cache_operation(args.read_cache, "read")
                print("    Data shape: {} Cells X {} Genes".format(*adata.X.shape), end=', ')
                try:
                    adata.obs["label"]
                    print("labels are available")
                except KeyError:
                    print("labels are not available")
                print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(adata.X) / adata.X.size)))
                return adata
            else:
                raise FileNotFoundError("!!! Cache not found at {}".format(args.read_cache))

        # Read raw
        if os.path.isfile(CACHE_DATA_FULL_PATH):
            print("!!! Cache available at {}. ".format(CACHE_DATA_FULL_PATH) + "Consider reading cache directly")
        if os.path.isdir(args.read_raw):
            raise FileExistsError("!!! Expect a file while received a directory.")
        adata = create_anndata(args)

        return adata

    def _load_batch(args):
        if args.read_cache is not None:
            if os.path.isfile(args.read_cache):
                adata = cache_operation(args.read_cache, "read")
                print("    Data shape: {}".format(adata.X.shape), end=', ')
                try:
                    adata.obs["label"]
                    print("labels are available", end='')
                except AttributeError:
                    print("labels are not available", end='')
                try:
                    adata.obs["batch"]
                    print(", batch information is available")
                except AttributeError:
                    print("\n!!! You specified batch_correction flag but batch information is not available in cache.")
                print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(adata.X) / adata.X.size)))
                return adata
            else:
                raise FileNotFoundError("!!! Cache not found at {}".format(args.read_cache))
        else:
            if os.path.isfile(CACHE_DATA_FULL_PATH.split(".pkl")[0] + "_batch_" + args.batch_correction + ".pkl"):
                print("!!! Cache available at {}. ".format(CACHE_DATA_FULL_PATH.split(".pkl")[0] + "_batch_" + args.batch_correction + ".pkl") + "Consider reading cache directly")
            if not os.path.isdir(args.read_raw):
                print("!!! Flag batch_correction is specified but only a single file provided")
                return _load_data(args)
            print(">>> Loading batches from directory {}".format(args.read_raw))
            file_list = [file for file in os.listdir(args.read_raw) if file.lower().endswith((".txt", ".csv", ".tsv", ".xls", ".xlsx"))]
            batches = []
            path_original = args.read_raw
            for filename in file_list:
                print(">>> Reading {}/{} from {}".format(file_list.index(filename) + 1, len(file_list), filename))
                args.read_raw = os.path.join(path_original, filename)
                data_batch = create_anndata(args)
                data_batch.obs['batch_raw'] = filename.split(".")[0]
                batches.append(data_batch)
            print(">>> Concatenating data")
            adata = anndata.AnnData.concatenate(*batches, join=args.batch_correction, fill_value=0)
            print(">>> Post-concatenation processing")
            adata = normalize_batch(adata, batch="batch_raw")
            if args.batch_correction == "outer":
                adata.var['gene_symbols'] = adata.var_names
            adata.obs['label'] = label_str_to_int(None, None, adata.obs['label_raw'], LABEL_COL, False)
            print("    Concatenated shape: {} Cells X {} Genes".format(*adata.X.shape))
            print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(adata.X) / adata.X.size)))
        return adata

    adata = _load_data(args) if args.batch_correction == "none" else _load_batch(args)

    # Write cache
    if CACHE_WRITE and args.read_raw is not None:
        check_directory(CACHE_DIR)
        cache_path = CACHE_DATA_FULL_PATH if args.batch_correction == "none" else \
            CACHE_DATA_FULL_PATH.split(".pkl")[0] + "_batch_" + args.batch_correction + ".pkl"
        if os.path.isfile(cache_path):
            while True:
                ans = input("!!! Cache {} exists. Confirm overwrite? [Y/N] ".format(cache_path))
                if ans.lower() == 'y':
                    os.remove(cache_path)
                    cache_operation(cache_path, "write", adata)
                    break
                elif ans.lower() == 'n':
                    print("    Cache NOT written")
                    break
                else:
                    print("!!! Invalid input")
        else:
            cache_operation(cache_path, "write", adata)

    return adata


def create_anndata(args):
    path = args.read_raw if args.read_raw is not None else args.read_cache
    print(">>> Reading data from {}".format(path))
    data = pd.read_csv(path, sep=args.sep, header=args.row_header - 1, index_col=0)
    data = data if not args.transpose else data.T
    data = data.iloc[:, args.col_header - 1:].astype(float)
    adata = AnnData(data)
    adata.var['gene_symbols'] = data.columns
    adata.obs['barcode'] = data.index
    print("    Data shape: {} Cells X {} Genes".format(*adata.X.shape))
    print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(adata.X) / adata.X.size)))

    if args.label is not None:
        if args.label.lower() != "barcode":
            print(">>> Reading labels from {}".format(args.label))
            if not os.path.isfile(args.label):
                raise FileNotFoundError("!!! Labels not found at {}".format(args.label))
            label_raw = pd.read_csv(args.label, sep=args.sep, usecols=[LABEL_COL]).to_numpy()
        else:
            print(">>> Use barcode information as label")
            label_raw = np.array([name.split('_')[0].upper() for name in adata.obs['barcode'].to_numpy()])
        try:
            label = label_raw.copy().astype(int)
        except ValueError:
            print(">>> Preprocessing labels")
            label = label_str_to_int(None, None, label_raw, None, preproc=False)
        label = np.squeeze(label)
        adata.obs['label'] = label
        adata.obs['label_raw'] = label_raw

        print("    Label Shape: {} with {} clusters: {}".format(label.shape, len(np.unique(label_raw)),
                                                                np.unique(label_raw)))
    return adata


def create_scdataset(args, self):
    self.path = args.read_raw
    self.row_header = args.row_header
    self.col_header = args.col_header
    self.transpose = args.transpose
    self.sep = args.sep
    self.label_dir = args.label
    self.device = "cuda" if args.cuda else "cpu"

    print(">>> Reading data from {}".format(self.path))
    self.data = pd.read_csv(self.path, sep=self.sep, header=self.row_header - 1)
    self.cell_name = np.array(self.data.columns)
    self.gene_name = np.squeeze(self.data.index)
    self.gene_filtered = self.gene_name
    self.data = self.data.iloc[:, self.col_header - 1:].astype(float)
    self.data = self.data.to_numpy() if not self.transpose else self.data.to_numpy().T
    print("    Data shape: {} Cells X {} Genes".format(*self.data.shape))
    print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(self.data) / self.data.size)))
    self.dim = self.data.shape[1]

    # Sanity check
    assert self.cell_name.shape[0] == self.data.shape[0]
    assert self.gene_name.shape[0] == self.data.shape[1]

    if self.label_dir is not None:
        if self.label_dir.lower() != "barcode":
            print(">>> Reading labels from {}".format(self.label_dir))
            if not os.path.isfile(self.label_dir):
                raise FileNotFoundError("!!! Labels not found at {}".format(self.label_dir))
            self.label_raw = pd.read_csv(self.label_dir, sep=self.sep, usecols=[LABEL_COL])
        else:
            print(">>> Use barcode information as label")
            self.label_raw = np.array([name.split('_')[0].upper() for name in self.cell_name])
        try:
            self.label = self.label_raw.copy().astype(int)
        except ValueError as err:
            print(">>> Preprocessing labels")
            self.label = label_str_to_int(None, None, self.label_raw, None, preproc=False)

        self.label = np.squeeze(self.label)
        # Sanity check
        assert self.data.shape[0] == self.label_raw.shape[0]
        assert self.data.shape[0] == self.label.shape[0]
        print("    Label Shape: {} with {} clusters: {}".format(self.label.size, len(np.unique(self.label_raw)),
                                                                np.unique(self.label_raw)))


class SingleCellDataset(Dataset):
    """
    Single cell dataset
    """
    def __init__(self, adata):
        super(SingleCellDataset, self).__init__()
        self.data = adata.X
        self.dim = self.data.shape[1]
        self.gene = adata.var['gene_symbols'].to_numpy()
        self.barcode = adata.obs['barcode'].to_numpy()
        # Store label is available
        try:
            self.label_raw = adata.obs['label_raw'].to_numpy()
            self.label = adata.obs['label'].to_numpy()
            self.label_avail = True
        except KeyError:
            self.label_avail = False
        # Store batch info if batch correction is applied
        try:
            self.batch_id = adata.obs['batch'].to_numpy()
            self.batch_raw = adata.obs['batch_raw'].to_numpy()
        except KeyError:
            pass

    def __getitem__(self, index):
        return self.data[index] if not self.label_avail else (self.data[index], self.label[index])

    def __len__(self):
        return len(self.data)

    def update_pars(self):
        self.dim = self.data.shape[1]


def cast_dataset_loader(dataset, device, batch_size):
    dataset.update_pars()
    dataset.data = torch.tensor(dataset.data, device=device).float()
    if dataset.label_avail:
        dataset.label = torch.tensor(dataset.label, device=device).long()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def add_noise(adata, args):
    """
    Simulating dropout & gaussian noise
    :param dataset:       original data
    :param args:                    argparser
    :return: noisy data
    """
    def _add_dropout(data, ops):
        print(">>> Simulating dropout noise with prob={}".format(ops))
        mask = np.random.binomial(1, 1 - ops, size=data.shape)
        data *= mask
        return data

    def _add_gaussian(data, sig):
        print(">>> Simulating Gaussian noise with relative sigma={}".format(sig))
        mask = np.random.normal(loc=0, scale=sig, size=data.shape)
        mask = np.where(mask < -1, -1, mask)
        mask = np.where(mask > 1, 1, mask)
        mask += 1
        data *= mask
        return data.astype(np.int)

    np.random.seed(NUMPY_RAND_SEED)

    if args.noise == "d":
        adata.X = _add_dropout(adata.X, args.dropout)
    elif args.noise == "g":
        adata.X = _add_gaussian(adata.X, args.gaussian)
    elif args.noise == "dg":
        adata.X = _add_dropout(adata.X, args.dropout)
        adata.X = _add_gaussian(adata.X, args.gaussian)
    elif args.noise == 'n':
        pass
    else:
        raise NotImplementedError("!!! Invalid noise options")
    print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(adata.X) / adata.X.size)))
    return adata


def normalize_data(data, method="lognormal"):
    print(">>> Normalizing data")
    if method == "lognormal":
        data = np.log1p(data)
        data = data / np.max(data)
        mean, std = np.mean(data), np.std(data)
        data -= mean
        data /= std
    elif method == "log":
        data = np.log1p(data)
        data = data / np.max(data)
    elif method == "normal":
        mean, std = np.mean(data), np.std(data)
        data -= mean
        data /= std
    elif method == "normalp":
        mean, std = np.mean(data, axis=1, keepdims=True), np.std(data, axis=1, keepdims=True)
        data -= mean
        data /= std
    return data


def normalize_batch(adata, batch, thres=6):
    if batch in adata.obs.keys():
        df = pd.Series(adata.obs[batch],dtype="category")
        for batch_label in df.cat.categories:
            tmp = adata[df==batch_label]
            tmp = tmp.X.copy()
            tmp = normalize_data(tmp, "normal")
            if max_value is not None:
                tmp[tmp>max_value] = max_value
            adata.X[df==category] = tmp
    return adata


def filter_gene(dataset, args):
    """
    Filter low-expressed genes
    :param dataset: noisy dataset
    :param args: argparser
    :return: filtered dataset
    """

    if args.sd_filter != 1:
        print(">>> Filtering dataset by top {:.2f} % std".format(100 * args.sd_filter))
        if args.sd_filter > 1 or args.sd_filter < 0:
            raise ValueError("!!! Invalid parameter for sd filtering provided, [0, 1 (no filtering)] expected")

        gene_num = dataset.dim

        sd = np.std(dataset.data, axis=0)

        # Find top genes
        filter_idx = (-sd).argsort(kind="stable")[:int(dataset.data.shape[1] * args.sd_filter)]
        dataset.data = dataset.data[:, sorted(filter_idx)]
        dataset.dim = dataset.data.shape[1]
        dataset.update_gene_name(dataset.gene_name[sorted(filter_idx)])

        print("    {} ({:.2f}%) genes are filtered".format(gene_num - dataset.dim,
                                                           100 * (gene_num - dataset.dim) / gene_num))
        print("    Top 5 genes are: {}, {}, {}, {}, {}".format(dataset.gene_name[filter_idx[0]],
                                                               dataset.gene_name[filter_idx[1]],
                                                               dataset.gene_name[filter_idx[2]],
                                                               dataset.gene_name[filter_idx[3]],
                                                               dataset.gene_name[filter_idx[4]]))
        print("    New shape: {}".format(dataset.data.shape))
        print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(dataset.data) / dataset.data.size)))

    if args.mean_filter != 0:
        print(">>> Filtering dataset by mean threshold = {}".format(args.mean_filter))
        if args.mean_filter > 1 or args.mean_filter < 0:
            raise ValueError("!!! Invalid parameter for mean filtering provided, [0 (no filtering), 1] expected")
        gene_num = dataset.dim
        avg_exp_level = np.average(np.where(dataset.data > 0, 1, 0), axis=0) # Binary exp level
        filter_idx = np.where(avg_exp_level > args.mean_filter)[0]
        dataset.data = dataset.data[:, filter_idx]
        dataset.dim = dataset.data.shape[1]
        dataset.update_gene_name(dataset.gene_filtered[filter_idx])

        # Find top 5
        top_5_idx = (-avg_exp_level[filter_idx]).argsort(kind="stable")[:5]
        print("    {} ({:.2f}%) genes are filtered".format(gene_num - dataset.dim,
                                                           100 * (gene_num - dataset.dim) / gene_num))
        print("    Top 5 genes are: {}, {}, {}, {}, {}".format(dataset.gene_filtered[top_5_idx[0]],
                                                               dataset.gene_filtered[top_5_idx[1]],
                                                               dataset.gene_filtered[top_5_idx[2]],
                                                               dataset.gene_filtered[top_5_idx[3]],
                                                               dataset.gene_filtered[top_5_idx[4]]))
        print("    New shape: {}".format(dataset.data.shape))
        print("    Sparsity: {:.2f} %".format(100 * (1 - np.count_nonzero(dataset.data) / dataset.data.size)))

    return dataset


def check_directory(directory):
    """
    Check existence of directory, create (recursively) if not.
    :param directory: directory
    :return: none
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print("+++ Directory {} created".format(directory))


def cache_operation(directory, mode, data=None, text=''):
    """
    Cache operation with pickle
    :param directory: cache location
    :param mode: read/write
    :param data: data for write
    :param text: notification text
    :return: data if load, none if write
    """
    if mode == "read":
        print(">>> Loading" + text + " cache from {}".format(directory))
        with open(directory, 'rb') as f:
            data = pkl.load(f)
        return data
    elif mode == "write":
        print(">>> Writing" + text + " cache to {}".format(directory))
        with open(directory, 'wb') as f:
            pkl.dump(data, f, protocol=4)
        return None
    else:
        raise ValueError("!!! Unsupported mode")


class Logger():
    def __init__(self, headline: list, log_name: str):
        self.log_length = len(headline)
        self.headline = '\t'.join(headline)
        self.log_name = log_name
        check_directory(LOG_PATH)
        self.f = self.init_file()
        self.log_count = 0
        self.log(self.headline)

    def init_file(self):
        return open(os.path.join(LOG_PATH, self.log_name), 'w')

    def log(self, info):
        assert len(info) == self.log_length
        self.f.writelines('\t'.join(info))
        self.log_count += 1

    def done(self):
        self.f.close()
        print("Logger {} closed; {} lines logged; file written to {}".format(
            Logger.__name__, self.log_count - 1, os.path.join(LOG_PATH, self.log_name)))
