import os
import numpy as np
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Set up data loaders
from datasets.datasets_analogy import AnalogyDataset, TestCarDataset
import argparse
import pandas as pd

# Set up the network and training parameters
from models.networks import AnalogyRelationNetwork
from metrics import AccumulatedAccuracyMetric


parser = argparse.ArgumentParser(description='Relation Network')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--cand_num', type=int, default=8)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save', type=str, default='./snapshots/relation_analogy_{}_{}/')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--vgg', action='store_true', default=False)
parser.add_argument('--mask_whole', action='store_true', default=False)
parser.add_argument('--another', action='store_true', default=False)
parser.add_argument('--iter', type=int, default=-1)

args = parser.parse_args()
m = 'vgg' if args.vgg else 'cnn'
args.save = args.save.format(m, args.cand_num)
os.makedirs(args.save, exist_ok=True)

torch.cuda.set_device(args.gpu)
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

test_dataset = TestCarDataset(test_path, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std),
                         ]), mask_whole=args.mask_whole, another=args.another)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

cuda = torch.cuda.is_available()
kwargs = {'num_workers': 16, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

model = AnalogyRelationNetwork(n_classes=1, vgg=args.vgg)
if args.resume > 0:
    checkpoint = torch.load(os.path.join(args.save, 'epoch_{}'.format(args.iter)))
    model.load_state_dict(checkpoint)
    print('loading model from {}'.format(os.path.join(args.save, 'epoch_{}'.format(args.iter))))
elif args.test:
    test_model = 'best_model' if args.iter < 0 else 'epoch_{}'.format(args.iter)
    checkpoint = torch.load(os.path.join(args.save, test_model), map_location='cpu')
    print('loading from {}'.format(os.path.join(args.save, test_model)))
    model.load_state_dict(checkpoint)
model.cuda()
loss_fn = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, int(0.4 * args.epochs), gamma=0.1, last_epoch=-1)
eval_epoch = 1
start_epoch = args.resume
log_interval = 1
metric = AccumulatedAccuracyMetric()


def preprocessing(images):
    # A B C Dx8
    B, n_cand, _, _, _ = images.shape
    data = torch.zeros((B, 2 * images.shape[-3], images.shape[-2], images.shape[-1])).to(images.device)
    data[:, :3] = images[:, 0]
    data[:, 3:] = images[:, 1]
    candidates = torch.zeros((B, n_cand - 3, 2 * images.shape[-3], images.shape[-2], images.shape[-1])).to(images.device)
    candidates[:, :, :3] = images[:, 2].unsqueeze(1).expand(-1, n_cand - 3, -1, -1, -1)
    candidates[:, :, 3:] = images[:, 3:]
    return data, candidates


def test(save=True):
    metric.reset()
    model.eval()
    part_correct = 0.0
    piece_correct = 0.0
    total = 0
    questions = []
    part_pred = []
    piece_pred = []
    part_question_acc = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    piece_question_acc = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    part_wrong_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    part_wrong_car_right_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    part_right_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}

    piece_wrong_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    piece_wrong_car_right_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    piece_right_car_wrong_kp = {'q12': [], 'q34': [], 'q56': [], 'q78': []}
    for batch_idx, (images, target, quest) in enumerate(testloader):
        assert len(images) == 1
        q = os.path.basename(quest[0])
        if cuda:
            images = images[0].cuda()
        # A, B1, B2, C, D1, D2, D3, D4
        data = torch.zeros((1, 2 * images.shape[-3], images.shape[-2], images.shape[-1])).to(images.device)
        data[0, :3] = images[0]
        candidates = torch.zeros((1, 4, 2 * images.shape[-3], images.shape[-2], images.shape[-1])).to(images.device)
        candidates[0, :, :3] = images[3].unsqueeze(0).expand(4, -1, -1, -1)
        candidates[0, :, 3:] = images[4:]

        target = target[0].max(1)[1]
        target = target.numpy()

        quest = quest[0].split('/')[-3:]
        quest = '/'.join(quest)
        questions.append(quest)
        # part
        data[0, 3:] = images[1]
        outputs = model(data, candidates)
        pred = outputs.data.max(1)[1].cpu()
        part_correct += (pred.numpy() == target[0]).sum()
        part_question_acc[q].append((pred.numpy() == target[0]).sum())
        part_right_car_wrong_kp[q].append((pred.numpy() == 1).sum())
        part_wrong_car_right_kp[q].append((pred.numpy() == 2).sum())
        part_wrong_car_wrong_kp[q].append((pred.numpy() == 3).sum())
        total += 1
        part_pred.append(pred.numpy()[0] + 1)

        # piece
        data[0, 3:] = images[2]
        outputs = model(data, candidates)
        pred = outputs.data.max(1)[1].cpu()
        piece_correct += (pred.numpy() == target[1]).sum()
        piece_question_acc[q].append((pred.numpy() == target[1]).sum())
        piece_right_car_wrong_kp[q].append((pred.numpy() == 0).sum())
        piece_wrong_car_right_kp[q].append((pred.numpy() == 3).sum())
        piece_wrong_car_wrong_kp[q].append((pred.numpy() == 2).sum())
        piece_pred.append(pred.numpy()[0] + 1)

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
              format(key,np.array(piece_question_acc[key]).mean(),
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

        print('saving volumes to {}'.format(os.path.join(args.save, 'part_mask_{}.csv'.format(args.mask_whole))))
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

        print('saving volumes to {}'.format(os.path.join(args.save, 'piece_mask_{}.csv'.format(args.mask_whole))))
        pd.DataFrame(data=save_data).to_csv(
            os.path.join(args.save, 'piece_mask_{}.csv'.format(args.mask_whole)), index=False)

    return (piece_correct + part_correct) / float(2 * total)

if args.test:
    test()
else:
    for epoch in range(0, start_epoch):
        scheduler.step()

    best_test_acc = 0
    best_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        # Train stage
        metric.reset()

        model.train()
        losses = []
        total_loss = 0

        for batch_idx, (images, target) in enumerate(train_loader):
            if cuda:
                images, target = images.cuda(), target.cuda()
            data, candidates = preprocessing(images)
            optimizer.zero_grad()
            outputs = model(data, candidates)
            target = target.max(1)[1]

            metric((outputs, ), (target, ))
            target = target.long()
            loss = loss_fn(outputs, target)
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        scheduler.step()
        total_loss /= (batch_idx + 1)

        if epoch % log_interval == 0:
            message = 'Training Relation Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch, args.epochs, total_loss)
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            print(message)
            torch.save(model.state_dict(), os.path.join(args.save, 'epoch_{}'.format(epoch + 1)))
            # Eval
            with torch.no_grad():
                metric.reset()
                model.eval()
                val_loss = 0
                for eval_i in range(eval_epoch):
                    for batch_idx, (images, target) in enumerate(val_loader):
                        if cuda:
                            images, target = images.cuda(), target.cuda()
                        data, candidates = preprocessing(images)
                        outputs = model(data, candidates)
                        target = target.max(1)[1]
                        metric((outputs, ), (target, ))

                        target = target.long()
                        loss = loss_fn(outputs, target)
                        val_loss += loss.item()

            val_loss /= len(val_dataset) * eval_epoch

            message = '======> Epoch: {}/{}. Validation average loss: {:.4f}'.format(epoch + 1, args.epochs, val_loss)
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            print(message)

            test_acc = test(save=False)
            if test_acc > best_test_acc:
                torch.save(model.state_dict(), os.path.join(args.save, 'best_model'))
                best_test_acc = test_acc
                best_epoch = epoch + 1
            print('----------------------------------------------------------------')

    print('best model achieved on epoch {} with testing accuracy {:.4f}'.format(best_epoch, best_test_acc))