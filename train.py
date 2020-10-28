#   -*- coding: utf-8 -*-
#
#   train.py
#   
#   Developed by Tianyi Liu on 2020-05-26 as tianyi
#   Copyright (c) 2020. All Rights Reserved.

"""

"""


import time
import argparse
import torch


from cfg import *
from analyze import run_dr, plot_embedding, run_optics
from utils import load_data, SingleCellDataset, normalize_data, add_noise, cast_dataset_loader
from model import SAE, AE
from eval import SAELoss, cal_ari, cast_tensor


LABEL_AVAL = True


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
    parser.add_argument("--batch_correction",
                        dest="batch_correction",
                        type=str,
                        default="none",
                        choices=["inner", "outer"],
                        help="Batch correction; outer: keep the union of all genes; inner: keep same genes.")
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
                        default=100,
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
        print("{:18}".format(item), "->\t", args.__dict__[item])
    return args


def visualize_results(loader, model, epoch, args):
    """
    Visualize intermediate results
    :param loader: dataloader
    :param model: nn model
    :param epoch: current epoch
    :param args: argparser
    :return: loss
    """
    model.eval()
    # Iterate through all data
    datas, nn_embedding, labels, loss = [], [], [], 0
    with torch.no_grad():
        for step, data_batch in enumerate(loader):
            if LABEL_AVAL:
                (data, label) = data_batch
                labels.extend(label.detach().cpu().numpy())
            else:
                (data) = data_batch
                labels = None
            y, mu, h1, y1 = model(data)
            datas.extend(data.detach().cpu().numpy())
            nn_embedding.extend(mu.detach().cpu().numpy())
            loss_w, loss_pca = compute_loss(data, y, mu, h1, y1)
            loss += (loss_w * LOSS_W_WG + loss_pca * LOSS_PCA_WG) * len(data)
        # Average loss among all data
        loss /= len(datas)
        # Run T-SNE embedding
        if (epoch + 1) == VISUL_EPOCH and args.dr_label:
            embedding = run_dr(datas, "TSNE", args=args)
            plot_embedding(embedding, label=labels) if LABEL_AVAL else plot_embedding(embedding)

        cluster_embedding(nn_embedding, labels, epoch, dr_type="TSNE")


def cluster_embedding(nn_embedding, labels=None, epoch=None, dr_type="TSNE"):
    """
    Clustering with nn embedding
    :param nn_embedding: nn embedding
    :param labels: labels, if available
    :param epoch:
    :param dr_type:
    :return:
    """
    embedding = run_dr(nn_embedding, "TSNE", epoch=epoch)
    pred = run_optics(embedding)

    # If label provided,
    if labels is None:
        plot_embedding(embedding, label=labels, epoch=epoch, dr_type="TSNE")
    else:
        print("    ARI: {}".format(cal_ari(pred, labels)))

    plot_embedding(embedding, pred, labels, epoch=epoch, dr_type=dr_type)


if __name__ == "__main__":
    tic = time.time()

    torch.manual_seed(TORCH_RAND_SEED)

    args = parse_args()

    device = "cuda" if args.cuda else "cpu"

    print('\n', " Loading Data ".center(50, "="), sep='')
    adata = load_data(args)
    adata.X = normalize_data(adata.X)
    clean_dataset = SingleCellDataset(adata)
    clean_loader = cast_dataset_loader(clean_dataset, device, args.batch_size)
    
    adata_noisy = add_noise(adata, args)
    adata_noisy.X = normalize_data(adata_noisy.X)
    noisy_dataset = SingleCellDataset(adata_noisy)
    noisy_loader = cast_dataset_loader(noisy_dataset, device, args.batch_size)


    """
    # Initial PCA
    if args.pca > 0:
        noisy_dataset.data = run_dr(noisy_dataset.data, "PCA", dim=args.pca)
        print("    Data shape: {}".format(noisy_dataset.data.shape))
    elif args.pca == 0:
        noisy_dataset.data = run_dr(noisy_dataset.data, "PCA", dim=PCA_DIM)
        print("    Data shape: {}".format(noisy_dataset.data.shape))
    elif args.pca < 0 and args.pca != -1:
        raise ValueError("!!! Invalid PCA DR parameter provided.")
    """

    toc_1 = time.time()

    print('\n', " Training Model ".center(50, "="), sep='')
    model = SAE([noisy_dataset.dim, 512, 128, 64], device).to(device)
    model.train_sub_ae(noisy_loader, args.lr, args.epoch)
    model.stack()
    
    # model = AE([noisy_dataset.dim, 512, 128, 64]).to(device)
    print(model)
    # criterion = SAELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # AE.fit(model, noisy_loader, optimizer, criterion, args.epoch)

    print(">>> Fine-tuning stacked auto-encoder")
    criterion = SAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 5)
    SAE.fit(model, noisy_loader, optimizer, criterion, args.epoch)

    toc_2 = time.time()

    sae_embedding = SAE.get_embedding(model, clean_loader)
    tsne_embedding = run_dr(cast_tensor(sae_embedding), dr_type="TSNE", cache=False)
    try:
        clean_dataset.batch_raw
        plot_embedding(tsne_embedding, label=clean_dataset.label_raw, batch_correction=clean_dataset.batch_raw, dr_type="TSNE")
    except KeyError:
        plot_embedding(tsne_embedding, label=clean_dataset.label_raw, dr_type="TSNE")

    toc_3 = time.time()

    print("Elapsed Time: {:.2f} s; Pre-proc: {:.2f} s; Training: {:.2f} s; Post-proc: {:.2f} s".format(toc_3 - tic,
                                                                                                       toc_1 - tic,
                                                                                                       toc_2 - toc_1,
                                                                                                       toc_3 - toc_2))



