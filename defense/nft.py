'''
Augmented Neural Fine-Tuning for Efficient Backdoor Purification

This file is modified based on the following source:
link : https://github.com/nazmul-karim170/NFT-Augmented-Backdoor-Purification

@inproceedings{karim2024augmented,
    title={Augmented Neural Fine-Tuning for Efficient Backdoor Purification},
    author={Karim, Nazmul and Al Arafat, Abdullah and Khalid, Umar and Guo, Zhishan and Rahnavard, Nazanin},
    booktitle={European Conference on Computer Vision},
    year={2024}
}

The defense method is called nft.

The update include:
    1. data preprocess and dataset setting
    2. model setting
    3. args and config
    4. save process
    5. new standard: robust accuracy
basic sturcture for defense method:
    1. basic setting: args
    2. attack result(model, train data, test data)
    3. nft defense:
        a. get some clean data
        b. replace each BatchNorm2d with a masked BatchNorm (per-channel mask, init 1)
        c. train only the masks with MixUp + L1 mask regularization, projecting the
           masks onto [mu(l), 1] after each gradient step
        d. fold the masks back into the BatchNorm weights
    4. test the result and get ASR, ACC, RC
'''

import argparse
import math
import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
from tqdm import tqdm

from defense.base import defense

from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import (
    get_input_shape, get_num_classes, get_transform,
)
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.choose_index import choose_index
from utils.trainer_cls import given_dataloader_test


class MaskedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d with a trainable per-channel mask: effective weight = weight * neuron_mask."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = nn.Parameter(torch.ones(num_features))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        bn_training = True if self.training else \
            (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * self.neuron_mask, self.bias,
            bn_training, exponential_average_factor, self.eps,
        )

    def to_standard_bn(self) -> nn.BatchNorm2d:
        """Fold the mask into gamma and return a standard BatchNorm2d."""
        bn = nn.BatchNorm2d(self.num_features, eps=self.eps, momentum=self.momentum,
                            affine=self.affine, track_running_stats=self.track_running_stats)
        bn.weight.data = (self.weight.data * self.neuron_mask.data).clone()
        bn.bias.data = self.bias.data.clone()
        bn.running_mean.data = self.running_mean.data.clone()
        bn.running_var.data = self.running_var.data.clone()
        bn.num_batches_tracked.data = self.num_batches_tracked.data.clone()
        bn.eval()
        return bn


def _set_module_by_name(model, name, new_module):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def replace_bn_with_masked(model):
    """Replace every BatchNorm2d with MaskedBatchNorm2d, copying its state."""
    masked = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d) and not isinstance(module, MaskedBatchNorm2d):
            mbn = MaskedBatchNorm2d(module.num_features, module.eps, module.momentum,
                                    module.affine, module.track_running_stats)
            mbn.load_state_dict(module.state_dict(), strict=False)
            mbn.neuron_mask.data.fill_(1.0)
            mbn.to(module.weight.device)
            _set_module_by_name(model, name, mbn)
            masked.append((name, mbn))
    return masked


def fold_masks_back(model, masked_layers):
    """Fold the masks and restore standard BatchNorm2d layers."""
    for name, mbn in masked_layers:
        _set_module_by_name(model, name, mbn.to_standard_bn())


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mask_clip(mask_params, mask_alpha, mask_beta, upper=1.0):
    """Project each mask onto [mu(l), 1], mu(l) = mask_alpha * exp(-mask_beta * l)."""
    with torch.no_grad():
        for count_layer, m in enumerate(mask_params, start=1):
            m.clamp_(mask_alpha * math.exp(-mask_beta * count_layer), upper)


class nft(defense):
    r"""Basic class for nft defense method."""

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)
        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        args.__dict__ = defaults

        args.terminal_info = sys.argv
        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"
        self.args = args

        if 'result_file' in args.__dict__ and args.result_file is not None:
            self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])

        parser.add_argument('--checkpoint_load', type=str)
        parser.add_argument('--checkpoint_save', type=str)
        parser.add_argument('--log', type=str)
        parser.add_argument("--dataset_path", type=str)
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str, help='the location of result')

        parser.add_argument('--nb_epochs', type=int, help='total number of gradient steps')
        parser.add_argument('--epoch_aggregation', type=int, help='gradient steps per outer pass')
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--eta_min', type=float, help='eta_min of CosineAnnealingLR')
        parser.add_argument('--model', type=str, help='resnet18')
        parser.add_argument('--random_seed', type=int)
        parser.add_argument('--yaml_path', type=str, default="./config/defense/nft/config.yaml")

        parser.add_argument('--ratio', type=float, help='ratio of clean training data')
        parser.add_argument('--index', type=str, help='index of clean data')
        parser.add_argument('--mixup_alpha', type=float, help='alpha of the MixUp Beta(alpha, alpha)')
        parser.add_argument('--eta_c', type=float, help='coefficient of the L1 mask regularizer')
        parser.add_argument('--mask_alpha', type=float, help='mask_clip: mu(l) = mask_alpha * exp(-mask_beta * l)')
        parser.add_argument('--mask_beta', type=float, help='mask_clip: per-layer floor decay')
        parser.add_argument('--save_name', type=str, help='output subfolder under defense/ (default: nft)')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        subdir = getattr(self.args, 'save_name', None) or 'nft'
        save_path = 'record/' + result_file + f'/defense/{subdir}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not os.path.exists(self.args.checkpoint_save):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not os.path.exists(self.args.log):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()
        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))
        try:
            logging.info(pformat(get_git_info()))
        except Exception:
            logging.info('Getting git info fails.')

    def set_devices(self):
        self.device = self.args.device

    def eval_step(self, model, clean_test_loader, bd_test_loader):
        criterion = nn.CrossEntropyLoss()
        device = self.device
        ca = given_dataloader_test(model, clean_test_loader, criterion, self.args.non_blocking, device)[0]['test_acc']
        asr = given_dataloader_test(model, bd_test_loader, criterion, self.args.non_blocking, device)[0]['test_acc']
        bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = True
        ra = given_dataloader_test(model, bd_test_loader, criterion, self.args.non_blocking, device)[0]['test_acc']
        bd_test_loader.dataset.wrapped_dataset.getitem_all_switch = False
        return ca, asr, ra

    def _train_one_pass(self, model, mask_params, n_per_class, train_loader, optimizer, criterion):
        args = self.args
        model.train()
        loss_sum, total = 0.0, 0
        for x, y, *_ in train_loader:
            x = x.to(args.device, non_blocking=args.non_blocking)
            y = y.to(args.device, non_blocking=args.non_blocking).long()
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup_alpha)
            optimizer.zero_grad()
            pred = model(mixed_x)
            ce = mixup_criterion(criterion, pred, y_a, y_b, lam)
            l1 = sum(torch.sum(torch.abs(1.0 - m)) for m in mask_params)
            loss = ce + (args.eta_c / max(n_per_class, 1)) * l1
            loss.backward()
            optimizer.step()
            mask_clip(mask_params, args.mask_alpha, args.mask_beta)
            loss_sum += loss.item()
            total += 1
        return loss_sum / max(total, 1)

    def mitigation(self):
        args = self.args
        self.set_devices()
        fix_random(args.random_seed)

        model = generate_cls_model(args.model, args.num_classes)
        model.load_state_dict(self.result['model'])
        model.to(args.device)

        # a. get some clean data
        train_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        ran_idx = choose_index(args, len(clean_dataset))
        np.savetxt(args.log + 'index.txt', ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        n_clean = len(ran_idx)
        n_per_class = n_clean / args.num_classes
        data_set_o = self.result['clean_train']
        data_set_o.wrapped_dataset = clean_dataset
        data_set_o.wrap_img_transform = train_tran
        # resampling with replacement, as in the official code
        sampler = torch.utils.data.RandomSampler(
            data_set_o, replacement=True,
            num_samples=args.epoch_aggregation * args.batch_size)
        trainloader = torch.utils.data.DataLoader(
            data_set_o, batch_size=args.batch_size, num_workers=args.num_workers,
            sampler=sampler, shuffle=False, pin_memory=args.pin_memory)

        test_tran = get_transform(args.dataset, *([args.input_height, args.input_width]), train=False)
        bd_testset = self.result['bd_test']
        bd_testset.wrap_img_transform = test_tran
        bd_test_loader = torch.utils.data.DataLoader(
            bd_testset, batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=False, shuffle=False, pin_memory=args.pin_memory)
        clean_testset = self.result['clean_test']
        clean_testset.wrap_img_transform = test_tran
        clean_test_loader = torch.utils.data.DataLoader(
            clean_testset, batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=False, shuffle=False, pin_memory=args.pin_memory)

        ca0, asr0, ra0 = self.eval_step(model, clean_test_loader, bd_test_loader)
        logging.info(f"[NFT] Before defense -> CA={ca0:.4f}  ASR={asr0:.4f}  RA={ra0:.4f}")

        # b. replace BatchNorm layers with masked BatchNorm and freeze everything else
        masked_layers = replace_bn_with_masked(model)
        model.to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        mask_params = []
        for _, mbn in masked_layers:
            mbn.neuron_mask.requires_grad = True
            mask_params.append(mbn.neuron_mask)
        logging.info(f"[NFT] {len(mask_params)} masked BatchNorm layers; "
                     f"N_c (samples/class)={n_per_class:.2f}")

        # c. train only the masks; nb_epochs counts gradient steps, so the outer loop
        # runs ceil(nb_epochs / epoch_aggregation) passes and the scheduler steps once per pass
        nb_iterations = int(math.ceil(args.nb_epochs / args.epoch_aggregation))
        optimizer = optim.SGD(mask_params, lr=args.lr, momentum=args.sgd_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=nb_iterations, eta_min=args.eta_min)
        criterion = nn.CrossEntropyLoss()

        agg_rows = []
        for it in tqdm(range(1, nb_iterations + 1)):
            lr_it = optimizer.param_groups[0]['lr']
            train_loss = self._train_one_pass(model, mask_params, n_per_class, trainloader, optimizer, criterion)
            scheduler.step()
            ca, asr, ra = self.eval_step(model, clean_test_loader, bd_test_loader)
            mask_l1 = float(sum(torch.sum(torch.abs(1.0 - m)) for m in mask_params).item())
            step = it * args.epoch_aggregation
            logging.info(f"[NFT] iter {it}/{nb_iterations} (step~{step}, lr={lr_it:.4f})  "
                         f"loss={train_loss:.4f}  CA={ca:.4f}  ASR={asr:.4f}  RA={ra:.4f}  ||1-m||1={mask_l1:.2f}")
            agg_rows.append({"iteration": it, "step": step, "lr": lr_it, "train_loss": train_loss,
                             "test_acc": ca, "test_asr": asr, "test_ra": ra, "mask_l1": mask_l1})
            pd.DataFrame(agg_rows).to_csv(args.save_path + "nft_df.csv", index=False)

        # d. fold the masks back into the BatchNorm weights
        fold_masks_back(model, masked_layers)
        model.to(args.device)
        ca1, asr1, ra1 = self.eval_step(model, clean_test_loader, bd_test_loader)
        logging.info(f"[NFT] After defense  -> CA={ca1:.4f}  ASR={asr1:.4f}  RA={ra1:.4f}")

        result = {'model': model}
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        return self.mitigation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    nft.add_arguments(parser)
    args = parser.parse_args()
    method = nft(args)
    if "result_file" not in args.__dict__ or args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = method.defense(args.result_file)
