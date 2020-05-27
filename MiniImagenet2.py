import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import pandas as pd

class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, n_way, k_shot, k_query, resize, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.mode = mode
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([
                                                 #lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([
                                                 #lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')
        self.lines = pd.read_csv(os.path.join(root, mode + '.csv'))
        print(len(self.lines))
        csvdata = self.loadCSV()
        self.images = [[]]*len(csvdata)
        for i, (k, v) in enumerate(csvdata.items()):
            # [[im11,im12,...],[im21,im22,...],[imn1,imn2,...]]
            for j,name in enumerate(v): # load all images
                img = Image.open(os.path.join(self.path, name)).resize((84,84)).convert('RGB')
                if j==0:
                    self.images[i] = [img]
                else:
                    self.images[i].append(img)
        self.cls_num = len(csvdata)

    def loadCSV(self):
        dictLabels = {}
        for i in range(len(self.lines)):
            filename = self.lines.iloc[i,0]
            label = self.lines.iloc[i,1]
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
        return dictLabels


    def __len__(self):
        return 12345678

    def __getitem__(self, index):
        selected_cls = np.random.choice(self.cls_num, self.n_way, False)
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_y = torch.zeros(self.setsz, dtype=torch.int64)
        meta_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        meta_y = torch.zeros(self.querysz, dtype=torch.int64)
        sup_pos = 0
        meta_pos = 0
        for idx, cls in enumerate(selected_cls):
            selected_imgs_idx = np.random.choice(len(self.images[cls]), self.k_shot + self.k_query, False)
            for i in range(self.k_shot):
                img = self.images[cls][selected_imgs_idx[i]]
                support_x[sup_pos] = self.transform(img)
                support_y[sup_pos] = idx
                sup_pos += 1
            for i in range(self.k_shot, self.k_shot + self.k_query):
                img = self.images[cls][selected_imgs_idx[i]]
                meta_x[meta_pos] = self.transform(img)
                meta_y[meta_pos] = idx
                meta_pos += 1
        return (support_x, support_y, meta_x, meta_y)
