import argparse
import os
import torch
from tqdm import tqdm
from pathlib import Path
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
import torch
import random
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from PIL import ImageFilter
import glob
import pandas as pd
from utils import load_state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeters(object):
    def __init__(self, size):
        self.meters = [AverageMeter(i) for i in range(size)]

    def update(self, idxs, vals):
        for i, v in zip(idxs, vals):
            self.meters[i].update(v)

    def get_avgs(self):
        return np.array([m.avg for m in self.meters])

    def get_sums(self):
        return np.array([m.sum for m in self.meters])

    def get_cnts(self):
        return np.array([m.count for m in self.meters])


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class test_Dataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.fnames = glob.glob(os.path.join(root, '*'))
        self.fnames.sort()
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname).convert('RGB')
        img = self.transform(img)
        label = 0
        id = fname.split('/')[-1].split('.')[0]
        return img, label, id

    def __len__(self):
        return self.num_samples


class FoodLTDataLoader(DataLoader):
    """
    ImageNetLT Data Loader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True):
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if training:
            dataset = datasets.DatasetFolder(os.path.join(data_dir, 'train'), loader=lambda x: Image.open(
                x), extensions="jpg", transform=train_trsfm)
            val_dataset = datasets.DatasetFolder(
                os.path.join(data_dir, 'val'), loader=lambda x: Image.open(x), extensions="jpg", transform=test_trsfm)
        else:  # test
            print(os.path.join(data_dir, 'test'))
            dataset = test_Dataset(os.path.join(data_dir, 'test'), test_trsfm)
            train_dataset = test_Dataset(
                os.path.join(data_dir, 'test'), TwoCropsTransform(train_trsfm))
            val_dataset = test_Dataset(
                os.path.join(data_dir, 'test'), test_trsfm)

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.n_samples = len(self.dataset)

        print(
            "Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            # 'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        # Note that sampler does not apply to validation set
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs)


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    load_state_dict(model, state_dict)
    model = model.to(device)
    weight_record_list = []
    data_loader = FoodLTDataLoader(
        config['data_loader']['args']['data_dir'],
        batch_size=16,
        shuffle=False,
        training=False,
        num_workers=0,
    )

    train_data_loader = data_loader.train_set()
    # valid_data_loader = data_loader.test_set()
    num_classes = config._config["arch"]["args"]["num_classes"]
    aggregation_weight = torch.nn.Parameter(
        torch.FloatTensor(3), requires_grad=True)
    aggregation_weight.data.fill_(1/3)

    optimizer = config.init_obj('optimizer', torch.optim, [aggregation_weight])

    for k in range(config["epochs"]):
        weight_record = test_training(
            train_data_loader, model, aggregation_weight, optimizer, num_classes, config, args)
        if weight_record[0] < 0.05 or weight_record[1] < 0.05 or weight_record[2] < 0.05:
            break
    torch.save({'weight': weight_record}, 'aggregation_weight.pth')
    print("Aggregation weight: Expert 1 is {0:.2f}, Expert 2 is {1:.2f}, Expert 3 is {2:.2f}".format(
        weight_record[0], weight_record[1], weight_record[2]))
    weight_record_list.append(weight_record)
    print('Aggregation weights of three experts:')
    for txt in weight_record_list:
        print(*txt)


def test_training(train_data_loader, model,  aggregation_weight, optimizer,   num_classes, config, args):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_data_loader),
        [losses])

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for i, (data, _, id) in enumerate(tqdm(train_data_loader)):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        output0 = model(data[0])
        output1 = model(data[1])
        expert1_logits_output0 = output0['logits'][:, 0, :]
        expert2_logits_output0 = output0['logits'][:, 1, :]
        expert3_logits_output0 = output0['logits'][:, 2, :]
        expert1_logits_output1 = output1['logits'][:, 0, :]
        expert2_logits_output1 = output1['logits'][:, 1, :]
        expert3_logits_output1 = output1['logits'][:, 2, :]
        aggregation_softmax = torch.nn.functional.softmax(
            aggregation_weight)  # softmax for normalization
        aggregation_output0 = aggregation_softmax[0].cuda() * expert1_logits_output0 + aggregation_softmax[1].cuda(
        ) * expert2_logits_output0 + aggregation_softmax[2].cuda() * expert3_logits_output0
        aggregation_output1 = aggregation_softmax[0].cuda() * expert1_logits_output1 + aggregation_softmax[1].cuda(
        ) * expert2_logits_output1 + aggregation_softmax[2].cuda() * expert3_logits_output1
        softmax_aggregation_output0 = F.softmax(aggregation_output0, dim=1)
        softmax_aggregation_output1 = F.softmax(aggregation_output1, dim=1)

        # SSL loss: similarity maxmization
        cos_similarity = cos(softmax_aggregation_output0,
                             softmax_aggregation_output1).mean()
        ssl_loss = cos_similarity
        loss = - ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(ssl_loss, data[0].shape[0])

    aggregation_softmax = torch.nn.functional.softmax(
        aggregation_weight, dim=0).detach().cpu().numpy()

    return np.round(aggregation_softmax[0], decimals=2), np.round(aggregation_softmax[1], decimals=2), np.round(aggregation_softmax[2], decimals=2)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--epochs', default=1, type=int,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    main(config)
