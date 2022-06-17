import os

import torch
import torch.nn as nn
import torch.utils.data
import yaml

import datasets
import models
import utils


class Builder:
    def __init__(self):
        self.config = {}
        self.model_name = ''

        # Load configs
        with open(os.path.join('configs', 'main.yaml')) as f:
            main = yaml.safe_load(f)
        with open(os.path.join('configs', main['config'])) as f:
            dataset_with_model = yaml.safe_load(f)
        with open(os.path.join('configs', 'train.yaml')) as f:
            train = yaml.safe_load(f)
        self.config.update(main)
        self.config.update(dataset_with_model)
        self.config.update(train)
        self.model_name = self.config['model_name']

    def build_dataset(self, dataset_type: str, ddp_enabled=False) -> tuple[torch.utils.data.Dataset,
                                                                           torch.utils.data.DataLoader]:
        cfg_dataset = self.config['dataset']

        root = cfg_dataset['root']
        num_workers = cfg_dataset['num_workers']
        batch_size = self.config[self.model_name]['batch_size']

        if dataset_type == 'train':
            transforms = datasets.transforms.Transforms(cfg_dataset['normalize_mean'],
                                                        cfg_dataset['normalize_std'],
                                                        self.config[self.model_name]['augmentation'])
            shuffle = True
            pin_memory = cfg_dataset['pin_memory']
        else:
            transforms = datasets.transforms.Transforms(cfg_dataset['normalize_mean'], cfg_dataset['normalize_std'])
            shuffle = False
            pin_memory = False

        # Dataset
        if cfg_dataset['name'] == 'CelebAMaskHQ':
            dataset = datasets.celebamaskhq.CelebAMaskHQ(root, dataset_type, transforms=transforms)
        elif cfg_dataset['name'] == 'Lane':
            dataset = datasets.lane.Lane(root, dataset_type, transforms=transforms)
        elif cfg_dataset['name'] == 'NIA':
            dataset = datasets.nia.NIA(root, dataset_type, transforms=transforms)
        else:
            raise ValueError('Wrong dataset name.')

        # Dataloader
        if ddp_enabled:
            sampler = torch.utils.data.DistributedSampler(dataset)
            shuffle = False
            pin_memory = False
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                                 num_workers=num_workers, pin_memory=pin_memory)
        return dataset, dataloader

    def build_model(self, num_classes: int, pretrained=False) -> nn.Module:
        cfg_model = self.config[self.model_name]

        if self.model_name == 'BiSeNet':
            model = models.bisenet.BiSeNet(num_classes)
        elif self.model_name == 'DeepLabV3':
            model = models.deeplabv3.DeepLabV3(cfg_model['backbone'], num_classes)
        elif self.model_name == 'EAGR':
            model = models.eagr.EAGR(num_classes)
        elif self.model_name == 'RegSeg':
            model = models.regseg.RegSeg(num_classes)
        elif self.model_name == 'SFNet':
            model = models.sfnet.sfnet_impl(cfg_model['backbone'], num_classes)
        elif self.model_name == 'UNet':
            model = models.unet.UNet(num_classes)
        else:
            raise ValueError('Wrong model name.')

        if pretrained:
            pretrained_weight_path = cfg_model['pretrained_weight']
            if os.path.isfile(pretrained_weight_path):
                state_dict = torch.load(pretrained_weight_path)
                model.load_state_dict(state_dict)
            else:
                print(f'FileNotFound: pretrained_weight ({self.model_name})')
        return model

    def build_criterion(self, device: torch.device) -> nn.Module:
        cfg_criterion = self.config[self.model_name]['criterion']

        class_weight = None
        if self.config['dataset']['class_weight'] is not None:
            class_weight = torch.tensor(self.config['dataset']['class_weight'], device=device)

        if cfg_criterion['name'] == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(class_weight, label_smoothing=cfg_criterion['label_smoothing'])
        elif cfg_criterion['name'] == 'FocalLoss':
            criterion = utils.loss.FocalLoss(alpha=cfg_criterion['alpha'], gamma=cfg_criterion['gamma'],
                                             label_smoothing=cfg_criterion['label_smoothing'])
        else:
            raise ValueError('Wrong criterion name.')
        return criterion

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg_optimizer = self.config[self.model_name]['optimizer']

        if cfg_optimizer['name'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg_optimizer['lr'], momentum=cfg_optimizer['momentum'],
                                        weight_decay=cfg_optimizer['weight_decay'], nesterov=cfg_optimizer['nesterov'])
        elif cfg_optimizer['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         cfg_optimizer['lr'],
                                         weight_decay=cfg_optimizer['weight_decay'])
        elif cfg_optimizer['name'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(),
                                          cfg_optimizer['lr'],
                                          weight_decay=cfg_optimizer['weight_decay'])
        elif cfg_optimizer['name'] == 'RAdam':
            optimizer = torch.optim.RAdam(model.parameters(),
                                          cfg_optimizer['lr'],
                                          weight_decay=cfg_optimizer['weight_decay'])
        elif cfg_optimizer['name'] == 'NovoGrad':
            optimizer = utils.optimizer.NovoGrad(model.parameters(),
                                                 cfg_optimizer['lr'],
                                                 weight_decay=cfg_optimizer['weight_decay'])
        else:
            raise ValueError('Wrong optimizer name.')
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg_scheduler = self.config[self.model_name]['scheduler']
        total_iters = self.config[self.model_name]['epoch']

        if cfg_scheduler['name'] == 'ConstantLR':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1, total_iters)
        elif cfg_scheduler['name'] == 'CosineLR':
            scheduler = utils.lr_scheduler.CosineLR(optimizer, total_iters)
        elif cfg_scheduler['name'] == 'LinearLR':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, total_iters)
        elif cfg_scheduler['name'] == 'PolyLR':
            scheduler = utils.lr_scheduler.PolyLR(optimizer, total_iters, cfg_scheduler['power'])
        else:
            raise ValueError('Wrong scheduler name.')
        return scheduler

    def build_aux_factor(self) -> list:
        cfg_aux_factor = self.config[self.model_name]['aux_factor']
        return cfg_aux_factor
