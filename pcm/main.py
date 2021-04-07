import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms
import deeplabplus
from syncar import SYNCARSegmentation
from utils import AverageMeter, inter_and_union

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='syncars',
                    help='name of dataset')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.007,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resize', type=int, default=800,
                    help='image resize size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--vis_pred', action='store_true', default=False,
                    help='visualize prediction result')
parser.add_argument('--pred_save_dir', type=str, default='results/pred/',
                    help='path to prediciton save dir')
parser.add_argument('--num_subtypes', type=int, default=0,
                    help='number of subtypes of object')
args = parser.parse_args()


def main():
  assert torch.cuda.is_available()
  torch.backends.cudnn.benchmark = True
  model_fname = 'results/models/deeplab_{0}_{1}_{2}_epoch%d.pth'.format(
      args.backbone, args.dataset, args.exp)
  if args.dataset == 'syncars':
    dataset = SYNCARSegmentation('data/syncars', 
                                 train=args.train, crop_size=args.crop_size,
                                 resize=args.resize, subtypes=args.num_subtypes>0)
  elif args.dataset == 'syncars_a':
    # data/syncars/analogy_qestions
    dataset = SYNCARSegmentation('data/analogy_question', 
                                 train=args.train, crop_size=None,
                                 resize=None, gt=False, subtypes=args.num_subtypes>0)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
    
  if args.backbone == 'resnet101':
    model = getattr(deeplabplus, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_subtypes=args.num_subtypes)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))

  if args.train:
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    criterion2 = nn.CrossEntropyLoss()
    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
      for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
          m.eval()
          m.weight.requires_grad = False
          m.bias.requires_grad = False
    backbone_params = (
        list(model.module.conv1.parameters()) +
        list(model.module.bn1.parameters()) +
        list(model.module.layer1.parameters()) +
        list(model.module.layer2.parameters()) +
        list(model.module.layer3.parameters()) +
        list(model.module.layer4.parameters()))
    last_params = list(model.module.aspp.parameters()) + list(model.module.decoder.parameters())
    if args.num_subtypes>0:
        last_params = list(model.module.aspp.parameters()) + \
            list(model.module.decoder.parameters()) + \
            list(model.module.classifier.parameters())
    optimizer = optim.SGD([
      {'params': filter(lambda p: p.requires_grad, backbone_params)},
      {'params': filter(lambda p: p.requires_grad, last_params)}],
      lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.train,
        pin_memory=True, num_workers=args.workers, drop_last=True)
    max_iter = args.epochs * len(dataset_loader)
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    start_epoch = 0

    if args.resume:
      if os.path.isfile(args.resume):
        print('=> loading checkpoint {0}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
      else:
        print('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
      for i, inp_tuple in enumerate(dataset_loader):
        if args.num_subtypes>0:
            inputs, target, subtype = inp_tuple
            subtype = Variable(subtype.long().cuda())
        else:
            inputs, target = inp_tuple
            
        cur_iter = epoch * len(dataset_loader) + i
        lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr * args.last_mult

        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
        
        outputs = model(inputs)
        if args.num_subtypes>0:
            o,s = outputs
            s = (s.squeeze(2)).squeeze(2)
            loss1 = criterion(o, target)
            loss2 = criterion2(s,subtype)
            loss = loss1 + 0.1*loss2
        else:
            loss = criterion(outputs, target)
        if np.isnan(loss.item()) or np.isinf(loss.item()):
          pdb.set_trace()
        losses.update(loss.item(), args.batch_size)
        if args.num_subtypes>0:
            losses1.update(loss1.item(), args.batch_size)
            losses2.update(loss2.item(), args.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('epoch: {0}\t'
              'iter: {1}/{2}\t'
              'lr: {3:.6f}\t'
              'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
              epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))
        if args.num_subtypes>0:
            print('loss1: {loss1.val:.4f} ({loss1.ema:.4f})\t'
                  'loss2: {loss2.val:.4f} ({loss2.ema:.4f})'.format(
                  loss1=losses1, loss2=losses2))

      if epoch % 10 == 9:
        torch.save({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, model_fname % (epoch + 1))

  else:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    model.eval()

    if os.path.isfile(args.resume):
      print('=> loading checkpoint {0}'.format(args.resume))
      checkpoint = torch.load(args.resume)
      state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
      model.load_state_dict(state_dict)
      print('=> loaded checkpoint {0} (epoch {1})'.format(
          args.resume, checkpoint['epoch']))
    else:
      print('=> no checkpoint found at {0}'.format(args.resume))
        
    
    cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()
    cls_meter = AverageMeter()
    sub_rst = dict()
    for i in range(len(dataset)):
      if args.num_subtypes>0:
        inputs, target, subtype = dataset[i]
      else:
        inputs, target = dataset[i]
      inputs = Variable(inputs.cuda())
      outputs = model(inputs.unsqueeze(0))
      if args.num_subtypes>0:
        outputs,s = outputs
        _, sub_cls = torch.max(s, 1)
        sub_cls = sub_cls.data.cpu().numpy().squeeze()
        s = nn.functional.softmax(s,dim=1).data.cpu().numpy().squeeze()
        
      _, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
        
      
      if args.vis_pred:
        if dataset.gt:
          path_string = dataset.masks[i].replace(dataset.root, '')
        else:
          path_string = dataset.images[i].replace(dataset.root, '')
        folder_name = '/'.join([pp for pp in path_string.split('/')[:-1] if len(pp)>0])
        imname = os.path.basename(path_string)
        
        mask_pred = Image.fromarray(pred)
        mask_pred.putpalette(cmap)
        os.makedirs(os.path.join(args.pred_save_dir, folder_name), exist_ok=True)
        mask_pred.save(os.path.join(args.pred_save_dir, folder_name, imname))
        if args.num_subtypes>0:
          sub_rst[path_string]=s
        
      if (i+1)%100==0:
        print('eval: {0}/{1}'.format(i + 1, len(dataset)))
      
      if dataset.gt:
        mask = target.numpy().astype(np.uint8)
        inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
        inter_meter.update(inter)
        union_meter.update(union)
        if args.num_subtypes>0:
          cls_meter.update(float(subtype==sub_cls))

    if dataset.gt:
      iou = inter_meter.sum / (union_meter.sum + 1e-10)
      for i, val in enumerate(iou):
        print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
      print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))
      if args.num_subtypes>0:
        print('Mean Cls Accuracy: {0:.2f}'.format(cls_meter.avg * 100))
        
    if args.vis_pred and args.num_subtypes>0:
      np.save(os.path.join(args.pred_save_dir, 'cls_pred.npy'), sub_rst)


if __name__ == "__main__":
  main()
