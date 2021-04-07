from __future__ import print_function

import torch.utils.data as data
import os, pickle, torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import preprocess
import numpy as np

class SYNCARSegmentation(data.Dataset):
    CLASSES = [
      'background', 'back_bumper', 'back_left_door', 'back_left_wheel', 'back_left_window',
      'back_license_plate', 'back_right_door', 'back_right_wheel', 'back_right_window',
      'back_windshield', 'front_bumper', 'front_left_door', 'front_left_wheel',
      'front_left_window', 'front_license_plate', 'front_right_door', 'front_right_wheel',
      'front_right_window', 'front_windshield', 'hood', 'left_frame',
      'left_head_light', 'left_mirror', 'left_quarter_window', 'left_tail_light', 
      'right_frame', 'right_head_light', 'right_mirror', 'right_quarter_window', 
      'right_tail_light', 'roof', 'trunk'
    ]
    
    SUBTYPES=['6710c87e34056a29aa69dfdc5532bb13', '42af1d5d3b2524003e241901f2025878',
              '4ef6af15bcc78650bedced414fad522f', '473dd606c5ef340638805e546aa28d99',
              'bad0a52f07afc2319ed410a010efa019'] # sedan, truck, minivan, suv, wagon
    
#     SUBTYPES=['42af1d5d3b2524003e241901f2025878',
#               '4ef6af15bcc78650bedced414fad522f', 
#               '473dd606c5ef340638805e546aa28d99'] # truck, minivan, suv

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 crop_size=None, gt=True, resize=None, subtypes=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.crop_size = crop_size
        self.gt = gt
        self.resize = resize
        self.subtypes = subtypes

        if self.train:
            _list_f = os.path.join(self.root, 'file_list_train.txt')
        else:
            _list_f = os.path.join(self.root, 'file_list_val.txt')
        self.images = []
        self.masks = []
        with open(_list_f, 'r') as lines:
            for line in lines:
                _image = os.path.join(self.root, line.split()[0])
                assert os.path.isfile(_image)
                self.images.append(_image)
                if gt:
                    _mask = os.path.join(self.root, line.split()[1])
                    assert os.path.isfile(_mask), _mask
                    self.masks.append(_mask)
                

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if not self.gt:
            _target = _img.split()[0]
            _input, _ = preprocess(_img, _target,
                               flip=False,
                               scale=(0.5, 1.0) if self.train else (0.5, 0.5),
                               crop=(self.crop_size, self.crop_size) if self.train else None, 
                               resize=None)
            
            if self.transform is not None:
                _input = self.transform(_input)
                
            if self.subtypes:
                return _input, None, None
                
            return _input, None
        
        if self.subtypes:
            _img_file_name = os.path.basename(self.images[index])
            _subtype_string = _img_file_name.split('_')[0]
            if _subtype_string in SYNCARSegmentation.SUBTYPES:
                _subtype = SYNCARSegmentation.SUBTYPES.index(_subtype_string)
            else:
                _subtype = None
            
        
        _target = Image.open(self.masks[index])
        # change small masks into 255
        _target_arr = np.array(_target)
        _th, _tw = _target_arr.shape[0], _target_arr.shape[1]
        _thres = int(_th * _tw * 0.01 * 0.01)
        _target_labels = np.unique(_target_arr)
        for _ll in _target_labels:
            if _ll == 0:
                continue
            if np.sum(_target_arr==_ll) < _thres:
                _target_arr[_target_arr==_ll]=255
                
        _target = Image.fromarray(_target_arr)
        
        _input, _target = preprocess(_img, _target,
                                   flip=False,
                                   scale=(0.5, 1.0) if self.train else (0.5,0.5),
                                   crop=(self.crop_size, self.crop_size) if self.train else None,
                                   resize=self.resize if self.train else None)
        
        if self.transform is not None:
            _input = self.transform(_input)

        if self.target_transform is not None:
            _target = self.target_transform(_target)
            
        if self.subtypes:
            return _input, _target, _subtype

        return _input, _target

    def __len__(self):
        return len(self.images)
    