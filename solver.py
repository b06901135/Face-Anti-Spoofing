import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
from sklearn.metrics import roc_auc_score

from model import *


class Solver():
    def __init__(self, args):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print(f'DEVICE: {self.device}')

        self.name = args.name

        self.step = 0
        self.epoch = 0
        self.warmup = args.warmup
        self.total_epoch = args.total_epoch
        self.checkpoint_epoch = args.checkpoint_epoch

        nets = {
            'resnet3d': ResNet3D,
            'resnet18': ResNet18,
            'resnet50': ResNet50,
            'alexnet': AlexNet
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

        self.writter = SummaryWriter(os.path.join('log', self.name))
        self.metrics = {
            'step': [],
            'train/loss': [],
            'train/acc':  [],
            'val/loss':   [],
            'val/acc':    []
        }

    def print_model(self):
        print(self.net)
        print(self.optimizer)

    def save_checkpoint(self, output_file, weights_only=False):
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'metrics': self.metrics,
            'net_state_dict': self.net.state_dict()
        }
        if not weights_only:
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict()
            })
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(checkpoint, output_file)

    def load_checkpoint(self, checkpoint_file, weights_only=True):
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        if not weights_only:
            self.step = checkpoint['step'] + 1
            self.epoch = checkpoint['epoch'] + 1
            self.metrics = checkpoint['metrics']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = checkpoint['step']

    def predict(self, loader):
        output = []
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                x = data.to(self.device)
                crop_num = x.size(1) if len(x.size()) == 5 else 1
                if crop_num > 1:
                    x = x.flatten(end_dim=1)

                y_pred = self.net(x)
                y_pred = F.softmax(y_pred, dim=1)

                if crop_num > 1:
                    y_pred = y_pred.view(-1, crop_num, y_pred.size(1))
                    y_pred = y_pred.mean(dim=1)

                output.extend(
                    y_pred[:, 0].detach().cpu().numpy()
                )

        return output

    def train_on_epoch(self, loader):
        loss = 0.0
        acc = 0.0
        # Calculate auc
        y_true = []
        y_score = []
        self.net.train()
        for i, data in enumerate(loader):
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

            message = f'epoch: {self.epoch + 1:>3} [{time.time() - start_time:2.2f}s] train [loss: {loss:.4f}, acc: {acc:.4f} auc: {auc:.4f}]'
            # Tensorboard
            self.writter.add_scalar('train/loss', loss, self.epoch + 1)
            self.writter.add_scalar('train/acc',  acc,  self.epoch + 1)
            self.writter.add_scalar('train/auc',  auc,  self.epoch + 1)
            # Checkpoint
            # self.metrics['train/loss'].append(loss)
            # self.metrics['train/acc'] .append(acc )

            if val_loader is not None:
                message += f' val [loss: {val_loss:.4f}, acc: {val_acc:.4f} auc: {val_auc:.4f}]'
                # Tensorboard
                self.writter.add_scalar('val/loss', val_loss, self.epoch + 1)
                self.writter.add_scalar('val/acc',  val_acc,  self.epoch + 1)
                self.writter.add_scalar('val/auc',  val_auc,  self.epoch + 1)
                # Checkpoint
                # self.metrics['val/loss'].append(val_loss)
                # self.metrics['val/acc'] .append(val_acc )

            print(message)

            if (self.epoch + 1) % self.checkpoint_epoch == 0:
                self.save_checkpoint(f'ckpt/{self.name}/e{self.epoch + 1:03}.pth')

            self.epoch += 1
