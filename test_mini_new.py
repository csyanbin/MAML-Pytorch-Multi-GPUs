import  torch, os
import  numpy as np
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import Meta
from    torch import optim
import plot
import json
import time
from collections import OrderedDict
from    copy import deepcopy
from dataset_mini import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=600000)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
argparser.add_argument('--imgc', type=int, help='imgc', default=3)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
argparser.add_argument('--weight_decay', type=float, default=1e-4)
argparser.add_argument('--gpu', type=str, default='0', help="gpu ids, default:0")
argparser.add_argument('--path', type=str, default='adam_clip', help="save path")
argparser.add_argument('--ckpt', type=str, default='net_119999_0.47866660356521606.pkl', help="checkpoint")
argparser.add_argument('--loader', type=int, default=0, help="0:default loader, 1:image all load, 2:pkl loader")
args = argparser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
n_gpus = len(args.gpu.split(','))

class Param:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = '/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch-Multi-GPUs/data/miniImagenet/'
    #out_path = '/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch-Multi-GPUs/ckpt/adam_clip/'
    out_path = '/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch-Multi-GPUs/ckpt/'+args.path+'/'
    #root = '/home/haoran/meta/miniimagenet/'
    #root = '/storage/haoran/miniimagenet/'
    #root = '/disk/0/storage/haoran/miniimagenet/'
    root = '/mnt/aitrics_ext/ext01/yanbin/MAML-Pytorch-Multi-GPUs/data/miniImagenet/'   #change to your own root!#
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

if not os.path.exists(Param.out_path):
    os.makedirs(Param.out_path)

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def inf_get(train):
    while (True):
        for x in train:
            yield x

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    #np.random.seed(222)
    test_result = {}
    best_acc = 0.0

    maml = Meta(args, Param.config).to(Param.device)
    if len(args.gpu.split(','))>1:
        maml = torch.nn.DataParallel(maml)
    state_dict = torch.load(Param.out_path+args.ckpt)
    print(state_dict.keys())
    pretrained_dict = OrderedDict()
    for k in state_dict.keys():
        if n_gpus==-1:
            pretrained_dict[k[7:]] = deepcopy(state_dict[k])
        else:
            pretrained_dict[k[0:]] = deepcopy(state_dict[k])
    maml.load_state_dict(pretrained_dict)
    print("Load from ckpt:", Param.out_path+args.ckpt)
    
    #opt = optim.Adam(maml.parameters(), lr=args.meta_lr)
    #opt = optim.SGD(maml.parameters(), lr=args.meta_lr, momentum=0.9, weight_decay=args.weight_decay)  

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    if args.loader in [0,1]: # default loader
        if args.loader==1:
            #from dataloader.mini_imagenet import MiniImageNet as MiniImagenet
            from MiniImagenet2 import MiniImagenet
        else:
            from MiniImagenet import MiniImagenet

        testset = MiniImagenet(Param.root, mode='test', n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, resize=args.imgsz)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)
        test_data = inf_get(testloader)

    elif args.loader==2: # pkl loader
        args_data = {}
        args_data['x_dim'] = "84,84,3"
        args_data['ratio'] = 1.0
        args_data['seed'] = 222
        loader_test  = dataset_mini(600, 100, 'test', args_data)
        loader_test.load_data_pkl()
 

    """Test for 600 epochs (each has 4 tasks)"""
    ans = None
    maml_clone = deepcopy(maml)
    for itr in range(600): # 600x4 test tasks
        if args.loader in [0,1]:
            support_x, support_y, qx, qy = test_data.__next__()
            support_x, support_y, qx, qy = support_x.to(Param.device), support_y.to(Param.device), qx.to(Param.device), qy.to(Param.device)
        elif args.loader==2:
            support_x, support_y, qx, qy = get_data(loader_test)
            support_x, support_y, qx, qy = support_x.to(Param.device), support_y.to(Param.device), qx.to(Param.device), qy.to(Param.device)

        temp = maml_clone(support_x, support_y, qx, qy, meta_train = False)
        if(ans is None):
            ans = temp
        else:
            ans = torch.cat([ans, temp], dim = 0)
        if itr%100==0:
            print(itr,ans.mean(dim = 0).tolist())
    meanacc = np.array(ans.mean(dim = 0).tolist())
    stdacc = np.array(ans.std(dim = 0).tolist())
    ci95 = 1.96*stdacc/np.sqrt(600)
    print(f'Acc: {meanacc[-1]:.4f}, ci95: {ci95[-1]:.4f}')
    with open(Param.out_path+'test.txt','w') as f:
        print(f'Acc: {meanacc[-1]:.4f}, ci95: {ci95[-1]:.4f}', file=f)

os.chdir(Param.out_path)
if __name__ == '__main__':
    main()

