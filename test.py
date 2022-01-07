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
from utils import load_state_dict, rename_parallel_state_dict


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
            train_dataset = datasets.DatasetFolder(os.path.join(data_dir, 'train'), loader=lambda x: Image.open(
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
        self.val = datasets.DatasetFolder(
            os.path.join('../final-project-challenge-3-tami/food_data/val'), loader=lambda x: Image.open(x), extensions="jpg", transform=test_trsfm)
        self.mapping = self.val.class_to_idx
        print(
            "Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")

        self.shuffle = False
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        # Note that sampler does not apply to validation set
        super().__init__(dataset=self.dataset, **self.init_kwargs)

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs), self.mapping


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def resume_checkpoint(resume_path, model, state_dict_only=True):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)

    state_dict = checkpoint['state_dict']
    if state_dict_only:
        rename_parallel_state_dict(state_dict)

    # self.model.load_state_dict(state_dict)
    load_state_dict(model, state_dict)


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    print(config.resume)
    state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    # load_state_dict(model, state_dict)
    rename_parallel_state_dict(state_dict)
    model.load_state_dict(state_dict)
    # model.load_state_dict([name.split('module.')[-1]
    #                       for name in state_dict.items()])

    # prepare model for testing
    model = model.to(device)
    weight_record_list = []
    data_loader = FoodLTDataLoader(
        '../final-project-challenge-3-tami/food_data',
        batch_size=128,
        shuffle=False,
        training=False,
        num_workers=0,
    )
    valid_data_loader, mapping = data_loader.test_set()
    # num_classes = config._config["arch"]["args"]["num_classes"]
    aggregation_weight = torch.nn.Parameter(
        torch.FloatTensor(3), requires_grad=False)
    aggregation_weight.data.fill_(1/3)
    mapping = {v: k for k, v in mapping.items()}
    checkpoint = torch.load('aggregation_weight.pth')
    aggregation_weight = checkpoint['weight']
    # aggregation_weight = torch.FloatTensor([0.6, 0.25, 0.15])
    print("Aggregation weight: Expert 1 is {0:.2f}, Expert 2 is {1:.2f}, Expert 3 is {2:.2f}".format(
        aggregation_weight[0], aggregation_weight[1], aggregation_weight[2]))
    test_validation(valid_data_loader, model,
                    aggregation_weight, device, mapping)


def test_validation(data_loader, model, aggregation_weight, device, mapping):
    model.eval()
    # aggregation_weight.requires_grad = False
    IDs = []
    prediction_results = []

    # total_logits = torch.empty((0, 1000)).cuda()
    # total_labels = []
    # acc = (prediction_results == total_labels).mean()
    with torch.no_grad():
        for i, (data, _, id) in enumerate(tqdm(data_loader)):
            # print(data.shape)
            data = data.to(device)
            output = model(data)
            expert1_logits_output = output['logits'][:, 0, :]
            expert2_logits_output = output['logits'][:, 1, :]
            expert3_logits_output = output['logits'][:, 2, :]
            aggregation_output = aggregation_weight[0] * expert1_logits_output + aggregation_weight[1] * \
                expert2_logits_output + \
                aggregation_weight[2] * expert3_logits_output
            prediction_results.extend(
                aggregation_output.argmax(axis=1).cpu().numpy().tolist())
            IDs.extend(id)
    head_df = ['image_id', 'label']
    df = pd.DataFrame(columns=head_df)
    for i in range(len(prediction_results)):
        df.loc[i] = [IDs[i]] + [int(mapping[prediction_results[i]])]
    df.to_csv('main_pred.csv', index=False)
    print('finish!')


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
