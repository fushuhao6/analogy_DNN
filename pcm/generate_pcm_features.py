import cv2
from scipy.io import loadmat
from PIL import Image
import glob,os
import numpy as np

data_dir = 'data/analogy_question/'
rst_dir = 'results/pred_AQ/'

cls_rst_file = os.path.join(rst_dir, 'cls_pred.npy')
cls_rst = np.load(cls_rst_file,allow_pickle=True).item()


cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
cmap = (cmap * 255).astype(np.uint8).flatten().tolist()
cls_names=['sedan', 'truck', 'minivan', 'suv', 'wagon']

acc_ls=[]
for QID in ['A','C']:
    img_ls = glob.glob(os.path.join(data_dir, '*','*','*','{}.png'.format(QID)))
    for img_i in range(len(img_ls)):
        img_file=img_ls[img_i]
        path_string = img_file.replace(data_dir,'')
        if QID == 'A':
            subtype_string = path_string.split('/')[1].split('_')[0]
        else:
            subtype_string = path_string.split('/')[1].split('_')[1]
        subtype_idx = cls_names.index(subtype_string)
        pred_cls = cls_rst['/'+path_string]
        acc_ls.append(float(np.argmax(pred_cls) == subtype_idx))
        
print('whole car image cls acc: {}'.format(np.mean(acc_ls)))
    
acc_ls=[]
for QID in ['B1','B2','D1','D2','D3','D4']:
    img_ls = glob.glob(os.path.join(data_dir, '*','*','*','{}.png'.format(QID)))
    for img_i in range(len(img_ls)):
        img_file=img_ls[img_i]
        path_string = img_file.replace(data_dir,'')
        if QID in ['B1','B2', 'D3','D4']:
            subtype_string = path_string.split('/')[1].split('_')[0]
        else:
            subtype_string = path_string.split('/')[1].split('_')[1]
        subtype_idx = cls_names.index(subtype_string)
        pred_cls = cls_rst['/'+path_string]
        acc_ls.append(float(np.argmax(pred_cls) == subtype_idx))
        
print('part/piece image cls acc: {}'.format(np.mean(acc_ls)))

pred_file_ls = list(cls_rst.keys())
pred_file_ls = sorted(pred_file_ls)
rst_dict=dict()
for kk in pred_file_ls:
    seg_rst=np.array(Image.open(os.path.join(rst_dir, kk[1:])))
    seg_his = np.zeros(31)
    for pp in range(31):
        seg_his[pp]=np.sum(seg_rst==pp+1)
        
    seg_his = seg_his/np.sum(seg_his)
    
    rst_dict[kk[1:]] = np.concatenate([seg_his, cls_rst[kk]])

with open(os.path.join(rst_dir, 'pcm_features.txt'),'w') as fh:
    for kk in pred_file_ls: 
        fh.write(kk[1:]+' '+' '.join([str(vv) for vv in rst_dict[kk[1:]]])+'\n')