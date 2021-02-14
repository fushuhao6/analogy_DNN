import os
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data import DataLoader
from models.networks import AnalogySiamese, AnalogySiameseVGG
import pandas as pd

import numpy as np
import argparse

# Set up data loaders
from datasets.datasets_analogy import AnalogyDataset, TestCarDataset

# Set up the network and training parameters
from losses import OnlineTripletLoss, ContrastiveLoss
from utils import AnalogyNegativeTripletSelector

parser = argparse.ArgumentParser(description='Siamese Analogy Network')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cand_num', type=int, default=4)
parser.add_argument('--save', type=str, default='./snapshots/siamese_analogy_{}_{}_{}/')
parser.add_argument('--method', type=str, choices=['concat', 'minus'], default='concat')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--model', type=str, default='vgg', choices=['resnet', 'vgg'])
parser.add_argument('--mask_whole', action='store_true', default=False)
parser.add_argument('--triplet', action='store_true', default=False)
parser.add_argument('--iter', type=int, default=-1)
parser.add_argument('--feat_dim', type=int, default=256)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.gpu)

cuda = torch.cuda.is_available()
mean, std = (0.45337084, 0.43081692, 0.40491408), (0.22491315, 0.2207595, 0.22327504)
test_mean, test_std = (0.58279017, 0.58185893, 0.57869234), (0.24194757, 0.24445643, 0.24695159)
root_path = '/ccvl/net/ccvl15/shuhao/datasets/analogy_questions'

train_list = os.path.join(root_path, 'train.lst')
val_list = os.path.join(root_path, 'test.lst')
test_path = '/ccvl/net/ccvl15/shuhao/datasets/question'

train_dataset = AnalogyDataset(root_path, list_file=train_list,
                             transform=transforms.Compose([
                                 transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std),
                             ]), mask_whole=args.mask_whole, cand_num=args.cand_num)
val_dataset = AnalogyDataset(root_path, list_file=val_list,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std),
                             ]), mask_whole=args.mask_whole, cand_num=args.cand_num)

trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

test_dataset = TestCarDataset(test_path, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(test_mean, test_std),
                         ]), mask_whole=args.mask_whole)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

args.save = args.save.format('triplet', args.cand_num, args.model) if args.triplet else args.save.format('pair', args.cand_num, args.model)
os.makedirs(args.save, exist_ok=True)
margin = 0.2
log_interval = 1
if args.model == 'vgg':
    model = AnalogySiameseVGG(feature_dim=args.feat_dim, method=args.method)
else:
    model = AnalogySiamese(feature_dim=args.feat_dim, method=args.method)

if args.test:
    test_model = 'best_model' if args.iter < 0 else 'epoch_{}'.format(args.iter)
    print('loading from {}'.format(os.path.join(args.save, test_model)))
    model.load_state_dict(torch.load(os.path.join(args.save, test_model), map_location=lambda storage, loc: storage))
elif args.resume > 0:
    print('loading model from {}'.format(os.path.join(args.save, 'epoch_{}'.format(args.resume))))
    model.load_state_dict(torch.load(os.path.join(args.save, 'epoch_{}.pth'.format(args.iter))))

if args.triplet:
    triplet_selector = AnalogyNegativeTripletSelector(margin=margin)
    loss_fn = OnlineTripletLoss(margin, triplet_selector)
else:
    contrast_loss = ContrastiveLoss(margin)
    def loss_fn(embeddings, target):
        B, embed_num, _ = embeddings.shape
        anchor = embeddings[:, 0].unsqueeze(1).repeat(1, embed_num-1, 1)
        candidates = embeddings[:, 1:].flatten(0, 1)

        anchor = anchor.flatten(0, 1)
        target = target.view(-1)
        loss = contrast_loss(anchor, candidates, target)

        sq_distances = (anchor - candidates).pow(2).sum(-1)     # B, 8
        positive = torch.argmax(target, dim=-1)
        pred = torch.argmin(sq_distances, dim=-1)
        correct = (pred == positive).sum()

        return loss, correct

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, int(0.4 * args.epochs), gamma=0.1, last_epoch=-1)

if args.cuda:
    model.cuda()


def processing(images, target):
    if args.cuda:
        images, target = images.cuda(), target.cuda()
    embeddings = model(images)      # B, 9, C
    loss, correct = loss_fn(embeddings, target)

    return loss, correct


def train(epoch):
    model.train()

    loss_all = 0.0
    correct = 0.0
    total = 0.0
    counter = 0
    for batch_idx, (images, target) in enumerate(trainloader):
        counter += 1
        loss, c = processing(images, target)
        correct += c
        total += target.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss

    acc = float(correct) / float(total) * 100.
    if epoch % log_interval == 0:
        print("Training Siamese Epoch {}: Avg Training Loss: {:.6f}, Acc: {:.4f}".format(epoch, loss_all/float(counter), acc))
    scheduler.step()
    return loss_all/float(counter), acc


def validate(epoch):
    model.eval()

    correct = 0.0
    total = 0.0
    loss_all = 0.0
    counter = 0
    with torch.no_grad():
        for e in range(1):
            for batch_idx, (images, target) in enumerate(valloader):
                counter += 1
                loss, c = processing(images, target)
                correct += c
                total += target.size(0)
                loss_all += loss.item()
        acc = float(correct) / float(total) * 100.
        if counter > 0:
            print("Epoch {}: Total Validation Loss: {:.6f}, Acc: {:.4f}".format(epoch, loss_all/float(counter), acc))
    return loss_all/float(counter), acc


def test(save=True):
    def testing(images, target):
        if args.cuda:
            images, target = images.cuda(), target.cuda()
        embeddings = model(images)  # B, 9, C
        B, n_emb, _ = embeddings.shape
        anchor = embeddings[:, 0].unsqueeze(1).repeat(1, n_emb - 1, 1)
        candidates = embeddings[:, 1:].flatten(0, 1)

        anchor = anchor.flatten(0, 1)
        target = target.view(-1)

        sq_distances = (anchor - candidates).pow(2).sum(-1)  # B, 8
        positive = torch.argmax(target, dim=-1)
        pred = torch.argmin(sq_distances, dim=-1)
        correct = (pred == positive).sum()

        return correct.data.cpu(), pred.data.cpu()

    model.eval()
    part_correct = 0.0
    piece_correct = 0.0
    total = 0.0
    part_question_acc = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    piece_question_acc = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    part_wrong_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    part_wrong_car_right_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    part_right_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}

    piece_wrong_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    piece_wrong_car_right_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    piece_right_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    with torch.no_grad():
        for batch_idx, (images, target, quest) in enumerate(testloader):
            assert len(images) == 1
            q = os.path.basename(quest[0])
            if cuda:
                images = images[0].cuda()
            # A, B1, B2, C, D1, D2, D3, D4
            data = torch.zeros(1, 7, 3, images.shape[-2], images.shape[-1])
            data[0, :2] = images[:2]
            data[0, 2:] = images[3:]

            # part
            c, pred = testing(data, target[0, 0].unsqueeze(0))
            part_correct += c
            part_question_acc[q].append(c)
            part_right_car_wrong_kp[q].append((pred.numpy() == 1).sum())
            part_wrong_car_right_kp[q].append((pred.numpy() == 2).sum())
            part_wrong_car_wrong_kp[q].append((pred.numpy() == 3).sum())

            # piece
            data[0, 1] = images[2]
            c, pred = testing(data, target[0, 1].unsqueeze(0))
            piece_correct += c
            piece_question_acc[q].append(c)
            piece_right_car_wrong_kp[q].append((pred.numpy() == 0).sum())
            piece_wrong_car_right_kp[q].append((pred.numpy() == 3).sum())
            piece_wrong_car_wrong_kp[q].append((pred.numpy() == 2).sum())
            total += 1
        print("Total Test Acc: {:.4f}, Part Acc: {:.4f}, Piece Acc: {:.4f}".format(
            (piece_correct + part_correct) / float(2 * total),
            part_correct / float(total), piece_correct / float(total)))

        for key in list(part_wrong_car_wrong_kp.keys()):
            print('{}: Part RCRK: {:.4f};\tPart RCWK: {:.4f};\tPart WCRK: {:.4f};\tPart WCWK: {:.4f}'.
                  format(key, np.array(part_question_acc[key]).mean(),
                         np.array(part_right_car_wrong_kp[key]).mean(),
                         np.array(part_wrong_car_right_kp[key]).mean(),
                         np.array(part_wrong_car_wrong_kp[key]).mean(), ))
            print('{}: Piece RCRK: {:.4f};\tPiece RCWK: {:.4f};\tPiece WCRK: {:.4f};\tPiece WCWK: {:.4f}'.
                  format(key, np.array(piece_question_acc[key]).mean(),
                         np.array(piece_right_car_wrong_kp[key]).mean(),
                         np.array(piece_wrong_car_right_kp[key]).mean(),
                         np.array(piece_wrong_car_wrong_kp[key]).mean(), ))

        if save:
            # part
            save_data = {' ': ['same, visible', 'same, invisible', 'different, visible', 'different, invisible'],
                         'correct': [], 'wrong kp': [], 'wrong car': [], 'wrong all': []}
            for key in list(part_wrong_car_wrong_kp.keys()):
                save_data['correct'].append(np.array(part_question_acc[key]).mean())
                save_data['wrong kp'].append(np.array(part_right_car_wrong_kp[key]).mean())
                save_data['wrong car'].append(np.array(part_wrong_car_right_kp[key]).mean())
                save_data['wrong all'].append(np.array(part_wrong_car_wrong_kp[key]).mean())

            print('saving volumes to {}'.format(
                os.path.join(args.save, 'part_mask_{}.csv'.format(args.mask_whole))))
            pd.DataFrame(data=save_data).to_csv(
                os.path.join(args.save, 'part_mask_{}.csv'.format(args.mask_whole)), index=False)

            # piece
            save_data = {' ': ['same, visible', 'same, invisible', 'different, visible', 'different, invisible'],
                         'correct': [], 'wrong kp': [], 'wrong car': [], 'wrong all': []}
            for key in list(part_wrong_car_wrong_kp.keys()):
                save_data['correct'].append(np.array(piece_question_acc[key]).mean())
                save_data['wrong kp'].append(np.array(piece_right_car_wrong_kp[key]).mean())
                save_data['wrong car'].append(np.array(piece_wrong_car_right_kp[key]).mean())
                save_data['wrong all'].append(np.array(piece_wrong_car_wrong_kp[key]).mean())

            print('saving volumes to {}'.format(
                os.path.join(args.save, 'piece_mask_{}.csv'.format(args.mask_whole))))
            pd.DataFrame(data=save_data).to_csv(
                os.path.join(args.save, 'piece_mask_{}.csv'.format(args.mask_whole)), index=False)

        return (piece_correct + part_correct) / float(2 * total)


def main():
    if args.test:
        test()
    else:
        if args.resume > 0:
            for e in range(args.resume):
                scheduler.step()
        best_test_acc = 0
        best_epoch = 0
        for epoch in range(args.resume, args.epochs):
            train(epoch)
            if (epoch + 1) % log_interval == 0:
                validate(epoch + 1)
                test_acc = test(save=False)
                if test_acc > best_test_acc:
                    torch.save(model.state_dict(), os.path.join(args.save, 'best_model'))
                    best_test_acc = test_acc
                    best_epoch = epoch + 1
                torch.save(model.state_dict(), args.save + 'epoch_{}.pth'.format(epoch + 1))
        print('best model achieved on epoch {} with testing accuracy {:.4f}'.format(best_epoch, best_test_acc))


if __name__ == '__main__':
    main()

