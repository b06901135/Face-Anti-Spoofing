import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
from sklearn.metrics import roc_auc_score

from model import *


class Solver():
    def __init__(self, args):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print(f'DEVICE: {self.device}')

        self.name = args.name
        self.texture = args.texture

        self.step = 0
        self.epoch = 0
        self.warmup = args.warmup
        self.total_epoch = args.total_epoch
        self.checkpoint_epoch = args.checkpoint_epoch

        nets = {
            'resnet3d': ResNet3D,
            'resnet18': ResNet18,
            'resnet50': ResNet50,
            'vgg11': Vgg11,
            'vgg16': Vgg16
        }

        try:
            self.net = nets[args.model]().to(self.device)
        except KeyError:
            raise NotImplementedError

        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)

        if self.warmup:
            l = lambda step: (step + 1) / 1000 if step < 1000 else 1.0
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, l)

        if args.load_checkpoint is not None:
            self.load_checkpoint(args.load_checkpoint, weights_only=False)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writter = SummaryWriter(os.path.join('log', self.name))
        except ModuleNotFoundError as e:
            print(e)
            self.writter = None

    def print_model(self):
        print(self.net)
        print(self.optimizer)

    def save_checkpoint(self, ckpt_file, weights_only=False):
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'net_state_dict': self.net.state_dict()
        }
        if not weights_only:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict()
            })

        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        torch.save(checkpoint, ckpt_file)

    def load_checkpoint(self, ckpt_file, weights_only=True):
        checkpoint = torch.load(ckpt_file, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if not weights_only:
            self.step = checkpoint['step'] + 1
            self.epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = checkpoint['step']

    def predict(self, loader, category=False):
        output = []
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                # print(f'predict: {i}/{len(loader)}')
                x = data.to(self.device)
                crop_num = x.size(1) if self.texture and len(x.size()) == 5 else 1
                if crop_num > 1:
                    x = x.flatten(end_dim=1)

                y_pred = self.net(x)
                y_pred = F.softmax(y_pred, dim=1)

                if crop_num > 1:
                    y_pred = y_pred.view(-1, crop_num, y_pred.size(1))
                    y_pred = y_pred.mean(dim=1)

                if category:
                    temp = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
                else:
                    temp = y_pred[:, 0].detach().cpu().numpy()
                output.extend(temp)

        return output

    def train_on_epoch(self, loader):
        loss = 0.0
        acc = 0.0
        # Calculate auc
        y_true = []
        y_score = []
        self.net.train()
        for i, data in enumerate(loader):
            # print(f'train: {i}/{len(loader)}')
            x = data[0].to(self.device)
            y = data[1].to(self.device)

            y_pred = self.net(x)
            batch_loss = F.cross_entropy(y_pred, y)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
            acc += (torch.argmax(y_pred.detach(), dim=1) == y).float().sum() / len(y)

            # Calculate auc
            y_true.extend((y == 0).detach().cpu().numpy())
            y_score.extend(F.softmax(y_pred, dim=1)[:, 0].detach().cpu().numpy())

            self.scheduler.step()
            self.step += 1

        return loss / len(loader), acc / len(loader), roc_auc_score(y_true, y_score)

    def val_on_epoch(self, loader):
        loss = 0.0
        acc = 0.0
        # Calculate auc
        y_true = []
        y_score = []
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                # print(f'val: {i}/{len(loader)}')
                x = data[0].to(self.device)
                y = data[1].to(self.device)

                y_pred = self.net(x)
                batch_loss = F.cross_entropy(y_pred, y)

                loss += batch_loss.item()
                acc += (torch.argmax(y_pred, dim=1) == y).float().sum() / len(y)

                # Calculate auc
                y_true.extend((y == 0).detach().cpu().numpy())
                y_score.extend(F.softmax(y_pred, dim=1)[:, 0].detach().cpu().numpy())

        return loss / len(loader), acc / len(loader), roc_auc_score(y_true, y_score)

    def train(self, train_loader, val_loader=None):
        while self.epoch < self.total_epoch:
            start_time = time.time()

            loss, acc, auc = self.train_on_epoch(train_loader)
            if val_loader is not None:
                val_loss, val_acc, val_auc = self.val_on_epoch(val_loader)

            message = f'epoch: {self.epoch + 1:>3} [{time.time() - start_time:7.2f}s] train [loss: {loss:.4f}, acc: {acc:.4f} auc: {auc:.4f}]'
            # Tensorboard
            if self.writter is not None:
                self.writter.add_scalar('train/loss', loss, self.epoch + 1)
                self.writter.add_scalar('train/acc',  acc,  self.epoch + 1)
                self.writter.add_scalar('train/auc',  auc,  self.epoch + 1)

            if val_loader is not None:
                message += f' val [loss: {val_loss:.4f}, acc: {val_acc:.4f} auc: {val_auc:.4f}]'
                # Tensorboard
                if self.writter is not None:
                    self.writter.add_scalar('val/loss', val_loss, self.epoch + 1)
                    self.writter.add_scalar('val/acc',  val_acc,  self.epoch + 1)
                    self.writter.add_scalar('val/auc',  val_auc,  self.epoch + 1)

            print(message)

            if (self.epoch + 1) % self.checkpoint_epoch == 0:
                self.save_checkpoint(f'ckpt/{self.name}/{self.epoch + 1:02}.pth')

            self.epoch += 1
