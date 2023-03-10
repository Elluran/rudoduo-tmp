import gc
from collections import defaultdict
from datetime import datetime

import torch
from torch import nn
from sklearn.metrics import f1_score
import numpy as np
from tqdm.notebook import tqdm
import mlflow


class Trainer:

    def loss_fn(self, logits, targets):
        criterion = nn.CrossEntropyLoss()
        return criterion(logits, targets)

    def __init__(self, device, model, train_dataloader, val_dataloader, optimizer, scheduler):
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = defaultdict(list)
        self.current_epoch = 0
        self.step = 0

    def train_epoch(self):
        self.model.train()
        losses = []
        for batch, idx in zip(tqdm(self.train_dataloader), range(len(self.train_dataloader))):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["labels"]
            outputs = self.model(input_ids=input_ids)
            loss = self.loss_fn(outputs, targets)
            losses.append(loss.item())
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            mlflow.log_metric("train loss", loss.item(), self.step)
            self.step += 1

            if idx % 20 == 0:
                print(np.mean(losses), loss.item())
            

        self.scheduler.step()
        return np.mean(losses)

    def eval_model(self):
        self.model.eval()

        true_labels = []
        predicted_labels = []

        losses = []
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["labels"]
                outputs = self.model(input_ids=input_ids)
                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())

                targets = targets.cpu()
                true_labels += list(targets[targets != -1])
                predicted_labels += nn.Softmax(dim=1)(outputs.cpu()).argmax(
                    dim=1).tolist()

        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_micro = f1_score(true_labels, predicted_labels, average='micro')
        f1_weighted = f1_score(
            true_labels, predicted_labels, average='weighted')

        return np.mean(losses), f1_macro, f1_micro, f1_weighted

    def train_loop(self, n_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(n_epochs):
            print(f'Epoch: {self.current_epoch}')
            print('-' * 10)
            

            train_loss = self.train_epoch()
            gc.collect()
            torch.cuda.empty_cache()

            print(f'Train loss: {round(train_loss, 4)}\n')
            torch.save(self.model.state_dict(
            ), f'./checkpoints/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}')

            val_loss, f1_macro, f1_micro, f1_weighted = self.eval_model()
            gc.collect()
            torch.cuda.empty_cache()

            print(f'Val loss: {round(val_loss, 4)}')
            print(f'F1 macro: {round(f1_macro, 4)}')
            print(f'F1 micro: {round(f1_micro, 4)}')
            print(f'F1 wghtd: {round(f1_weighted, 4)}\n')

            mlflow.log_metric("mean train loss", train_loss, self.current_epoch)
            mlflow.log_metric("val loss", val_loss, self.current_epoch)
            mlflow.log_metric("F1 macro", f1_macro, self.current_epoch)
            mlflow.log_metric("F1 micro", f1_micro, self.current_epoch)
            mlflow.log_metric("F1 wghtd", f1_weighted, self.current_epoch)
            self.current_epoch += 1

            self.history['val_loss'].append(val_loss)
            self.history['train_loss'].append(train_loss)
