import os
import torch
from os.path import join
import pickle
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_image(filename, opencv=False):
    if opencv:
        return cv2.imread(filename)
    else:
        return np.array(Image.open(filename))[:, :, :3]


def pad_image_list(im_list, pad_value=127):
    shapes = np.array([im.shape[:2] for im in im_list])
    max_shape = np.max(shapes, axis=0)
    padded_im_list = []
    for im in im_list:
        padded_im_list.append(pad_patch(max_shape, im, pad_value))
    return padded_im_list, max_shape


def pad_patch(expected_shape, patch, pad_value=127):
    pad_up = int((expected_shape[0] - patch.shape[0]) / 2)
    pad_down = int(expected_shape[0] - patch.shape[0] - pad_up)
    pad_left = int((expected_shape[1] - patch.shape[1]) / 2)
    pad_right = int(expected_shape[1] - patch.shape[1] - pad_left)

    assert pad_up >= 0 and pad_down >= 0 and pad_left >= 0 and pad_right >= 0, 'expected size {}, image size {}'.format(expected_shape, patch.shape )

    patch = np.pad(patch, ((pad_up, pad_down), (pad_left, pad_right), (0, 0)), 'constant', constant_values=pad_value)
    return patch


def resize_patch(im, patch):
    return cv2.resize(patch, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_AREA)


class AnalogyDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """
    def __init__(self, root_path, list_file, transform, mask_whole=False, cand_num=8):
        self.root_path = root_path
        with open(list_file, 'r') as f:
            analogy_list = f.readlines()
        self.analogy_list = [f.strip() for f in analogy_list]

        img_count = 0
        for f in self.analogy_list:
            img_count += len(os.listdir(join(self.root_path, f)))
        print('{} images in total.'.format(img_count))

        self.mask_whole = mask_whole
        self.transform = transform
        self.cand_num = cand_num        # number of candidate patches

        if mask_whole:
            print('Masking out whole images.')

        print('Read {} analogy questions.'.format(len(self.analogy_list)))

    def __getitem__(self, index):
        q_idx = self.analogy_list[index]

        part = torch.rand(1)[0] < 0.5
        label = np.zeros((self.cand_num, ), dtype=np.uint8)
        if part:
            B = read_image(join(self.root_path, q_idx, 'B1.png'))
            label[0] = 1
        else:
            B = read_image(join(self.root_path, q_idx, 'B2.png'))
            label[1] = 1
        # set image paths, A and C are whole images, B and D are patches
        if self.mask_whole:
            A = np.ones((512, 512, 3), dtype=np.uint8) * 127
            C = np.ones((512, 512, 3), dtype=np.uint8) * 127
        else:
            A = read_image(join(self.root_path, q_idx, 'A.png'))
            C = read_image(join(self.root_path, q_idx, 'C.png'))

        D_list = []
        for i in range(1, 100):
            if not os.path.exists(join(self.root_path, q_idx, 'D{}.png'.format(i))):
                break
            D_list.append(read_image(join(self.root_path, q_idx, 'D{}.png'.format(i))))

        assert self.cand_num <= len(D_list)

        # shuffle D list
        if self.cand_num < len(D_list):
            label = np.argmax(label)
            correct_D = D_list.pop(label)
            random.shuffle(D_list)
            D_list = [correct_D] + D_list[:self.cand_num - 1]
            label = np.zeros((len(D_list), ))
            label[0] = 1
        assert len(D_list) == len(label)
        c = list(zip(D_list, label))
        random.shuffle(c)
        D_list, label = zip(*c)

        # pad patches
        img_list = [A, B, C] + list(D_list)
        img_list, max_shape = pad_image_list(img_list)

        transformed_images = torch.zeros(len(img_list), 3, max_shape[0], max_shape[1])
        if self.transform is not None:
            for i in range(len(img_list)):
                transformed_images[i] = (self.transform(Image.fromarray(img_list[i])))
        return transformed_images, np.array(label)

    def __len__(self):
        return len(self.analogy_list)


class TestCarDataset(Dataset):
    def __init__(self, root_path, transform, mask_whole=False, another=False):
        self.root_path = root_path
        self.questions = []
        self.mask_whole = mask_whole
        self.another = another
        for kp in os.listdir(root_path):
            for car_type in os.listdir(os.path.join(root_path, kp)):
                for q in os.listdir(os.path.join(root_path, kp, car_type)):
                    cur_path = os.path.join(root_path, kp, car_type, q)
                    if len(os.listdir(cur_path)) == 8:
                        self.questions.append(cur_path)
                    else:
                        print('some files in {} is missing'.format(cur_path))

        self.transform = transform
        print('Read {} questions'.format(len(self.questions)))

    def __getitem__(self, index):
        q = self.questions[index]
        # read images, need whole image for image_a1 and image_b1, others are for patches.
        image_a = read_image(join(q, 'A.png'))
        part_a = read_image(join(q, 'B1.png'))
        piece_a = read_image(join(q, 'B2.png'))
        image_b = read_image(join(q, 'C.png'))

        if self.mask_whole:
            image_a = np.ones((image_a.shape[0], image_a.shape[1], 3), dtype=np.uint8) * 127
            image_b = np.ones((image_b.shape[0], image_b.shape[1], 3), dtype=np.uint8) * 127

        candidates = []
        if self.another:
            for i in range(2):
                candidates.append(np.array(Image.open(join(q, 'D{}.png'.format(int(i + 1))))))
            split_names = q.split('/')
            q_num = split_names[-1]
            car_types = split_names[-2].split('_')
            types = os.listdir(q + '/../../')
            candidate_types = []
            for t in types:
                is_in = False
                for c in car_types:
                    if c in t:
                        is_in = True
                if not is_in:
                    candidate_types.append(t)
            random.shuffle(candidate_types)
            t = candidate_types[0]
            new_q = os.path.join(q, '../../', t, q_num)
            for i in range(2, 4):
                candidates.append(np.array(Image.open(join(new_q, 'D{}.png'.format(int(i + 1))))))
        else:
            for i in range(4):
                candidates.append(np.array(Image.open(join(q, 'D{}.png'.format(int(i + 1))))))
        images = [image_a,  part_a, piece_a, image_b] + candidates

        images, _ = pad_image_list(images)

        targets = np.zeros((2, 4))
        targets[0, 0] = 1
        targets[1, 1] = 1
        images = np.array(images)

        transformed_images = torch.zeros((images.shape[0], images.shape[3], images.shape[1], images.shape[2]))
        if self.transform is not None:
            for i in range(len(images)):
                transformed_images[i] = (self.transform(Image.fromarray(images[i])))
        return transformed_images, targets, q

    def __len__(self):
        return len(self.questions)
