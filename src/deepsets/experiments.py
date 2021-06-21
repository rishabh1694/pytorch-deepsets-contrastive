import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from .datasets import MNISTSummation, MNIST_TRANSFORM
from .networks import InvariantModel, SmallMNISTCNNPhi, SmallRho
from torch.utils.data import DataLoader

from IPython import embed

import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE




class SumOfDigits(object):
    def __init__(self, lr=1e-3, wd=5e-3, batch_size=32, temp=0.5, k=10, length=10, dataset_len=1000):
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.temp = temp
        self.k = k
        self.c = 10
        self.length = length
        self.dataset_len = dataset_len
        # self.train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=100000, train=True, transform=MNIST_TRANSFORM)
        # self.test_db = MNISTSummation(min_len=5, max_len=50, dataset_len=100000, train=False, transform=MNIST_TRANSFORM)

        self.train_db = MNISTSummation(min_len=self.length, max_len=self.length, dataset_len=self.dataset_len, train=True, transform=MNIST_TRANSFORM)
        self.train_loader = DataLoader(self.train_db, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
        self.memory_db = MNISTSummation(min_len=self.length, max_len=self.length, dataset_len=self.dataset_len, train=True, transform=MNIST_TRANSFORM)
        self.memory_data_loader = DataLoader(self.memory_db, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True,
                              drop_last=True)
        self.test_db = MNISTSummation(min_len=self.length, max_len=self.length, dataset_len=self.dataset_len, train=False, transform=MNIST_TRANSFORM)
        self.test_data_loader = DataLoader(self.test_db, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)

        self.the_phi = SmallMNISTCNNPhi()
        # self.the_rho = SmallRho(input_size=10, output_size=1)
        self.the_rho = SmallRho(input_size=10, output_size=10)

        self.model = InvariantModel(phi=self.the_phi, rho=self.the_rho, length=self.length)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.summary_writer = SummaryWriter(
            log_dir='/home/souri/temp/deepsets/exp-lr_%1.5f-wd_%1.5f/' % (self.lr, self.wd))

    def train_1_epoch(self, epoch_num: int = 0):
        self.model.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
        # for i in tqdm(range(len(self.train_db))):
        for x1, x2, target in train_bar:
            if torch.cuda.is_available():
                x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()

            x1, x2, target = Variable(x1), Variable(x2), Variable(target)

            self.optimizer.zero_grad()
            feat1, out1 = self.model.forward(x1)
            feat2, out2 = self.model.forward(x2)

            # [2*B, D]
            out = torch.cat([out1, out2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temp)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

            # # the_loss = F.mse_loss(pred, target)
            # # print(pred, target)
            # the_loss = F.cross_entropy(pred, target)

            # compute loss
            pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / self.temp)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            loss.backward()
            self.optimizer.step()

            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size

        self.summary_writer.add_scalar('train_loss', total_loss / total_num, epoch_num)

    def evaluate(self):
        self.model.eval()
        total_top1, total_top5, total_num, feature_bank, targets = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(self.memory_data_loader, desc='Feature extracting'):
                feature, out = self.model(data)
                feature_bank.append(feature)
                targets.append(target)

            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            # feature_labels = torch.tensor(targets, device=feature_bank.device)
            feature_labels = torch.cat(targets, dim=0).t().contiguous().flatten()
            # loop test data to predict the label by weighted knn search
            test_bar = tqdm(self.test_data_loader)
            for data, _, target in test_bar:
                feature, out = self.model(data)

                total_num += data.size(0)
                # compute cos similarity between each feature vector and feature bank ---> [B, N]
                # print(feature.shape, feature_bank.shape)
                sim_matrix = torch.mm(feature, feature_bank)
                # [B, K]
                sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
                # [B, K]
                sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
                sim_weight = (sim_weight / self.temp).exp()

                # counts for each class
                one_hot_label = torch.zeros(data.size(0) * self.k, self.c, device=sim_labels.device)
                # [B*K, C]
                one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
                # weighted score ---> [B, C]
                pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, self.c) * sim_weight.unsqueeze(dim=-1), dim=1)

                pred_labels = pred_scores.argsort(dim=-1, descending=True)
                total_top1 += torch.sum((pred_labels[:, :1] == target).any(dim=-1).float()).item()
                total_top5 += torch.sum((pred_labels[:, :5] == target).any(dim=-1).float()).item()
                test_bar.set_description('Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                         .format(total_top1 / total_num * 100, total_top5 / total_num * 100))

        print(total_top1 / total_num * 100, total_top5 / total_num * 100)
        self.tSNE_vis(feature_labels,feature_bank,save_tag="tsne",save_figure=True)

    def tSNE_vis(self,
        targets,
        features,
        save_tag="",
        save_figure=False,
        feats_in_plot=50,
    ):
        """Plots the feature quality by the means of t-SNE
        Args:
            df: Dataframe
            features: Training instances
            class_labels: labels (strings)
            save_tag: title of plot to save
        Prints & Saves:
            t-SNE plot of 250 instances of each class
        """
        class_colours = ["green", "gray", "brown", "blue", "red", "black", "yellow", "orange", "pink", "violet"]
        class_instances = {}
        class_labels = np.arange(0,10)
        for i in class_labels:
            class_instances[i] = (targets == i).sum()

        tsne_m = TSNE(n_jobs=8, random_state=42)
        X_embedded = tsne_m.fit_transform(features.t())

        fig = plt.figure(figsize=(6, 6))
        lr = 150
        p = 50
        index = 0
        # PLOT
        for (label, colour, c_i) in zip(
            class_labels, class_colours, class_instances
        ):
            # indexes = self.random_indexes(
            #     index, index + class_instances[label], feats_in_plot
            # )
            idx = (targets == label).nonzero().flatten()
            indexes = np.random.choice(idx, size=feats_in_plot, replace=False)
            plt.scatter(X_embedded[indexes, 0], X_embedded[indexes, 1], c=colour)
            index += class_instances[label]

        fig.legend(
            bbox_to_anchor=(0.075, 0.061),
            loc="lower left",
            ncol=1,
            labels=class_labels,
        )
        if save_figure:
            plt.savefig(
                "../figures/" + save_tag + ".png", bbox_inches="tight",
            )

    def random_indexes(self, a, b, feats_in_plot):
        """Support function for tSNE_vis
        Args:
            a: start index
            b: end index
            feats_in_plot: # of featuers to be plotted per class
        Returns:
            Random list of feats_in_plot indexes between a and b
        """
        randomList = []
        # Set a length of the list to feats_in_plot
        for i in range(feats_in_plot):
            # any random numbers from a to b
            randomList.append(random.randint(a, b - 1))

        return randomList

        # self.model.eval()
        # totals = [0] * 51
        # corrects = [0] * 51

        # for i in tqdm(range(len(self.test_db))):
        #     x1, x2, target = self.test_db.__getitem__(i)

        #     item_size = x.shape[0]

        #     if torch.cuda.is_available():
        #         x = x.cuda()

        #     pred = self.model.forward(Variable(x)).data

        #     if torch.cuda.is_available():
        #         pred = pred.cpu().numpy().flatten()

        #     # pred = int(round(float(pred[0])))
        #     pred = int(round(float(torch.argmax(pred))))
        #     target = int(round(float(target.numpy()[0])))
        #     # print(pred,target)

        #     totals[item_size] += 1

        #     if pred == target:
        #         corrects[item_size] += 1

        # totals = np.array(totals)
        # corrects = np.array(corrects)
        # print(corrects,totals)
        # print(corrects / totals)
