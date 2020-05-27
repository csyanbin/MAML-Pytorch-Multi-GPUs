import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniImagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniImagenet/split')

class MiniImageNet(Dataset):
    """ Usage: 
    """
    def __init__(self, root, mode, n_way, k_shot, k_query, resize=84):
        self.mode = mode
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
 
        csv_path = osp.join(SPLIT_PATH, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.label = []
        lb = -1
        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                print(lb, end='\t')
            img = Image.open(path).resize((92, 92)).convert('RGB')
            self.data.append(img)
            self.label.append(lb)

        self.cls_num = len(set(self.label))

        # Transformation
        self.transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                np.array([0.229, 0.224, 0.225]))
        ])

        # class-id dict
        self.label = np.array(self.label)
        self.m_ind = []
        for i in range(max(self.label) + 1):
            ind = np.argwhere(self.label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        selected_cls = np.random.choice(self.cls_num, self.n_way, False)
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        support_y = torch.zeros(self.setsz, dtype=torch.int64)
        meta_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        meta_y = torch.zeros(self.querysz, dtype=torch.int64)
        sup_pos = 0
        meta_pos = 0
        for idx, cls in enumerate(selected_cls):
            selected_imgs_idx = np.random.choice(len(self.m_ind[cls]), self.k_shot+self.k_query, False)
            for i in range(self.k_shot):
                ii = cls*len(self.m_ind[cls])+selected_imgs_idx[i]
                img = self.data[ii]
                support_x[sup_pos] = self.transform(img)
                support_y[sup_pos] = idx
                sup_pos += 1
            for i in range(self.k_shot, self.k_shot + self.k_query):
                ii = cls*len(self.m_ind[cls])+selected_imgs_idx[i]
                img = self.data[ii]
                meta_x[meta_pos] = self.transform(img)
                meta_y[meta_pos] = idx
                meta_pos += 1
        return (support_x, support_y, meta_x, meta_y)
