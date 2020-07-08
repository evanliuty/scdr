#   -*- coding: utf-8 -*-
#
#   test.py
#   
#   Developed by Tianyi Liu on 2020-06-05 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import argparse


import numpy as np
from utils import load_data, filter_gene
from analyze import run_dr, plot_embedding


def parse_args():
    """
    Argparser
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--read_cache",
                       dest="read_cache",
                       help="Read data from cache.")
    group.add_argument("--read_raw",
                       dest="read_raw",
                       help="Read data from raw data.")
    parser.add_argument("--row_header",
                        dest="row_header",
                        type=int,
                        default=1,
                        help="# rows in header.")
    parser.add_argument("--col_header",
                        dest="col_header",
                        type=int,
                        default=1,
                        help="# column in header.")
    parser.add_argument("-t",
                        dest="transpose",
                        action="store_false",
                        help="Transpose data to shape (# cells, # genes); Default = TRUE.")
    parser.add_argument("--sep",
                        dest="sep",
                        default="\t",
                        help="Separator in data file.")
    parser.add_argument("--cuda",
                        dest="cuda",
                        action="store_false",
                        help="GPU Support; Default = TRUE.")
    parser.add_argument("--lr",
                        dest="lr",
                        type=float,
                        default=1e-2,
                        help="Initial learning rate.")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        default=128,
                        help="Batch size.")
    parser.add_argument("--epoch",
                        dest="epoch",
                        type=int,
                        default=200,
                        help="Training epoch.")
    parser.add_argument("--label",
                        dest="label",
                        help="Read label")
    parser.add_argument("--noise",
                        dest="noise",
                        default="n",
                        choices=["n", "d", "g", "dg"],
                        help="Noise simulation; n: none; d: dropout; g: gaussian.")
    parser.add_argument("--dropout",
                        dest="dropout",
                        type=float,
                        default=0,
                        help="Dropout probability.")
    parser.add_argument("--gaussian",
                        dest="gaussian",
                        type=float,
                        default=0,
                        help="Gaussian sigma.")
    parser.add_argument("--mean_filter",
                        dest="mean_filter",
                        type=float,
                        default=0,
                        help="Filter low-expressed genes by mean expression level.")
    parser.add_argument("--sd_filter",
                        dest="sd_filter",
                        type=float,
                        default=1,
                        help="Filter low-variant genes by standard deviation.")
    parser.add_argument("--pca",
                        dest="pca",
                        type=int,
                        default=-1,
                        help="Initial PCA DR: -1 -> None; 0 -> PCA_DIM in cfgs.py; +ve int -> CLI.")
    parser.add_argument("--dr_label",
                        dest="dr_label",
                        action="store_false",
                        help="Set -> DO NOT run DR for label (if available); Default = TRUE.")
    args = parser.parse_args()

    print('\n', " Call with Arguments ".center(50, "="), sep='')
    for item in args.__dict__:
        print("{:15}".format(item), "->\t", args.__dict__[item])
    return args


if __name__ == "__main__":
    args = parse_args()

    dataset = load_data(args)
    print(np.count_nonzero(dataset.data) / dataset.data.size)

