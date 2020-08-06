#   -*- coding: utf-8 -*-
#
#   preproc.py
#   
#   Developed by Tianyi Liu on 2020-06-10 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import os
import argparse
import numpy as np
import pandas as pd


from cfg import *


def parse_args():
    """
    Argparser
    :return: argparser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        dest="dir",
                        default="./placeholder",
                        help="Original directory")
    parser.add_argument("--sep",
                        dest="sep",
                        default="\t",
                        help="Separator")
    args = parser.parse_args()
    for item in args.__dict__:
        print("{:15}".format(item), "->\t", args.__dict__[item])
    return args


def label_str_to_int(directory, label_df, label_np, col, preproc=True):
    """
    String label to int
    :param directory: label directory
    :param label_df: label in pd dataframe
    :param label_np: label in np array
    :param col: col name in label_df
    :return: none
    """
    labels = np.unique(label_np)
    label_proc = label_np.copy()
    for index, item in enumerate(labels):
        label_proc[label_np == item] = index
    if preproc:
        col_name = col + "_proc" if col != "" else "proc"
        label_df.insert(label_df.shape[1], col_name, label_proc.astype(int), True)
        label_df.to_csv(directory.split('/')[-1].split('.')[0] + "_proc.csv", index=False, sep='\t')
        print(">>> Pre-processed file written to {}".format(directory.split('/')[-1].split('.')[0] + "_proc.csv"))
    else:
        return label_proc.astype(int)


def label_preproc(directory, sep='\t', col=LABEL_COL):
    """
    Pre-process labels if originally are strings
    :param directory: label directory
    :param sep: separator
    :param col: column name
    :return: none
    """
    label_df = pd.read_csv(directory, sep=sep)
    label_col = label_df[[col]]
    label_np = label_col.to_numpy()
    label_str_to_int(directory, label_df, label_np, col)


def barcode_label_preproc(directory, sep='\t', transpose=True):
    """
    Pre-process labels of humandata
    :param directory: label directory
    :param sep: separator
    :param transpose: transpose oringinal data to (# cells, # genes)
    :return: none
    """
    label_df = pd.read_csv(directory, sep=sep, nrows=1, header=None).transpose() if transpose else pd.read_csv(
        directory, sep=sep, nrows=1, header=None)
    label_np = np.squeeze(label_df.to_numpy())
    label_np = np.array([name.split('_')[0] for name in label_np])
    label_str_to_int(directory, label_df, label_np, "")


def extract_region(data, label, region, filename, region_col, cell_col):
    if region is None or len(region) == 0:
        region = pd.unique(label[region_col])
    print("Regions: {}".format(region))
    print("    Data shape: {}".format(data.shape))
    for step, reg in enumerate(region):
        print(">>> Extracting {} {}/{}".format(reg, step + 1, len(region)))
        lb = label[label[region_col] == reg]
        df = data[lb.index]
        print("   {} shape: {}".format(reg, df.shape))
        df.columns = lb[cell_col] + "_" + lb.index
        df.to_csv("./{}_{}.tsv".format(filename, reg), sep="\t")


if __name__ == "__main__":
    """
    DIRECTORY = "./placeholder"
    args = parse_args()
    if not os.path.isfile(args.dir):
        if not os.path.isfile(DIRECTORY):
            raise FileNotFoundError("!!! Cannot locate original label files at {} and {}".format(args.dir, DIRECTORY))
        else:
            print(">>> Reading labels from {}".format(DIRECTORY))
            human_label_preproc(DIRECTORY, args.sep)
    else:
        print(">>> Reading labels from {}".format(args.dir))
        human_label_preproc(args.dir)
    """


