import random
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def calculate_acc(output, target):
    pred = output.data.max(1)[1]
    target = target.data.view_as(pred)
    correct = pred.eq(target).cpu().sum().numpy()
    return correct * 100.0 / target.size()[0]


def calculate_correct(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct


def contrast_loss(output, target):
    gt_value = output
    noise_value = torch.zeros_like(gt_value)
    G = gt_value - noise_value
    zeros = torch.zeros_like(gt_value)
    zeros.scatter_(1, target.view(-1, 1), 1.0)
    return F.binary_cross_entropy_with_logits(G, zeros)


def reset_seed():
    seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    return seed


class AnalogyPairSelector:
    """
    Creates all possible positive pairs. For negative pairs, pairs with smallest distance are taken into consideration,
    matching the number of positive pairs.
    """

    def __init__(self, margin):
        super(AnalogyPairSelector, self).__init__()
        self.margin = margin

    def get_pairs(self, embeddings, labels):
        B, embed_num, _ = embeddings.shape
        anchor = embeddings[:, 0].unsqueeze(1).repeat(1, embed_num - 1, 1).detach()
        candidates = embeddings[:, 1:].detach()

        sq_distances = (anchor - candidates).pow(2).sum(-1)     # B, 8
        positive = torch.argmax(labels, dim=-1)
        pred = torch.argmin(sq_distances, dim=-1)
        correct = (pred == positive).sum()

        labels = labels.bool()
        dist_masked = sq_distances
        dist_masked[labels] = float('inf')
        hardest_negative = torch.argmin(dist_masked, dim=-1) + 1

        positive = positive.unsqueeze(1) + 1
        hardest_negative = hardest_negative.unsqueeze(1)
        positive_pairs = torch.cat((torch.zeros_like(positive), positive), dim=-1)      # B, 3
        negative_pairs = torch.cat((torch.zeros_like(positive), hardest_negative), dim=-1)      # B, 3

        residual = torch.arange(B) * embed_num
        residual = residual.unsqueeze(1).repeat(1, 2)
        positive_pairs = residual.to(positive_pairs.device) + positive_pairs
        negative_pairs = residual.to(negative_pairs.device) + negative_pairs

        return positive_pairs.long(), negative_pairs.long(), correct


class AnalogyNegativeTripletSelector:
    def __init__(self, margin):
        super(AnalogyNegativeTripletSelector, self).__init__()
        self.margin = margin

    def get_triplets(self, embeddings, labels):
        B, embed_num, _ = embeddings.shape
        anchor = embeddings[:, 0].unsqueeze(1).repeat(1, embed_num - 1, 1).detach()
        candidates = embeddings[:, 1:].detach()

        sq_distances = (anchor - candidates).pow(2).sum(-1)     # B, 8
        positive = torch.argmax(labels, dim=-1)
        pred = torch.argmin(sq_distances, dim=-1)
        correct = (pred == positive).sum()

        labels = labels.bool()
        dist_masked = sq_distances
        dist_masked[labels] = float('inf')
        hardest_negative = torch.argmin(dist_masked, dim=-1) + 1

        positive = positive.unsqueeze(1) + 1
        hardest_negative = hardest_negative.unsqueeze(1)
        triplets = torch.cat((torch.zeros_like(positive), positive, hardest_negative), dim=-1)      # B, 3

        residual = torch.arange(B) * embed_num
        residual = residual.unsqueeze(1).repeat(1, 3)
        triplets = residual.to(triplets.device) + triplets

        return triplets.long(), correct
