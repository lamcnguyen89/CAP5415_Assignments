"""

This module creates the training setup, configuration and validation

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch

import utils


# In this stage, we create an autoencoder and train it to learn the shape of a particular object. 
# Note that we are training one object type at a time.
class TrainerStage1:
    '''Train loop and evaluation for stage 1 Structure generator'''
    def __init__(self, cfg, data_loaders, criterions,
                 on_after_epoch=None, on_after_batch=None):
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.l1 = criterions[0]
        self.sigmoid_bce = criterions[1]
        self.iteration = 0
        self.epoch = 0
        self.history = []
        self.on_after_epoch = on_after_epoch
        self.on_after_batch = on_after_batch

    def train(self, model, optimizer, scheduler):
        print("======= TRAINING START =======")

        for self.epoch in range(self.cfg.startEpoch, self.cfg.endEpoch):
            print(f"Epoch {self.epoch}:")

            train_epoch_loss = self._train_on_epoch(model, optimizer)
            val_epoch_loss = self._val_on_epoch(model)

            hist = {
                'epoch': self.epoch,
                'train_loss_XYZ': train_epoch_loss["epoch_loss_XYZ"],
                'train_loss_mask': train_epoch_loss["epoch_loss_mask"],
                'train_loss': train_epoch_loss["epoch_loss"],
                'val_loss_XYZ': val_epoch_loss["epoch_loss_XYZ"],
                'val_loss_mask': val_epoch_loss["epoch_loss_mask"],
                'val_loss': val_epoch_loss["epoch_loss"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                images = self._make_images_board(model)
                self.on_after_epoch(model, pd.DataFrame(self.history),
                                    images, self.epoch)

        print("======= TRAINING DONE =======")
        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_XYZ = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for self.iteration, batch in enumerate(data_loader, self.iteration):
            input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)
            # ------ define ground truth------
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH), # [H,W]
                torch.arange(self.cfg.outW)]) # [H,W]
            XGT, YGT = XGT.float(), YGT.float()
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]), 
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0) #[2V,H,W]
            XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) # [1,2V,H,W] 

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
                mask = (maskLogit > 0).byte().to(torch.bool)
                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(depth.masked_select(mask),
                                    depthGT.masked_select(mask))
                loss_mask = self.sigmoid_bce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

                # ------ Update weights ------
                loss.backward()
                # True Weight decay
                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data.add_(param.data,
                                alpha = -self.cfg.trueWD * group['lr'])
                optimizer.step()

            if self.on_after_batch is not None:
                if self.cfg.lrSched.lower() in "cyclical":
                    self.on_after_batch(self.iteration)
                else: self.on_after_batch(self.epoch)

            running_loss_XYZ += loss_XYZ.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_XYZ = running_loss_XYZ / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tTrain loss: {epoch_loss}")
        return {"epoch_loss_XYZ": epoch_loss_XYZ,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _val_on_epoch(self, model):
        model.eval()

        data_loader = self.data_loaders[1]
        running_loss_XYZ = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0

        for batch in data_loader:
            input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)
            # ------ define ground truth------
            XGT, YGT = torch.meshgrid([
                torch.arange(self.cfg.outH), # [H,W]
                torch.arange(self.cfg.outW)]) # [H,W]
            XGT, YGT = XGT.float(), YGT.float()
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]), 
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0) #[2V,H,W]
            XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) # [1,2V,H,W] 

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN*3,:,:]
                mask = (maskLogit > 0).byte().to(torch.bool)
                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(depth.masked_select(mask),
                                    depthGT.masked_select(mask))
                loss_mask = self.sigmoid_bce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

            running_loss_XYZ += loss_XYZ.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_XYZ = running_loss_XYZ / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tVal loss: {epoch_loss}")
        return {"epoch_loss_XYZ": epoch_loss_XYZ,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _make_images_board(self, model):
        model.eval()
        num_imgs = 64

        batch = next(iter(self.data_loaders[1]))
        input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)

        with torch.set_grad_enabled(False):
            XYZ, maskLogit = model(input_images)
            XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
            depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
            mask = (maskLogit > 0).float()

        return {'RGB': utils.make_grid(input_images[:num_imgs]),
                'depth': utils.make_grid(1-depth[:num_imgs, 0:1, :, :]),
                'depth_mask': utils.make_grid(
                    ((1-depth)*mask)[:num_imgs, 0:1, :, :]),
                'depthGT': utils.make_grid(
                    1-depthGT[:num_imgs, 0:1, :, :]),
                'mask': utils.make_grid(
                    torch.sigmoid(maskLogit[:num_imgs, 0:1,:, :])),
                'maskGT': utils.make_grid(maskGT[:num_imgs, 0:1, :, :]),
                }

    def findLR(self, model, optimizer, writer,
               start_lr=1e-7, end_lr=10, num_iters=50):
        model.train()

        losses = []
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iters)

        for lr in lrs:
            # Update LR
            for group in optimizer.param_groups: group['lr'] = lr

            batch = next(iter(self.data_loaders[0]))
            input_images, depthGT, maskGT = utils.unpack_batch_fixed(batch, self.cfg.device)
            # ------ define ground truth------
            XGT, YGT = torch.meshgrid([torch.arange(self.cfg.outH), # [H,W]
                                       torch.arange(self.cfg.outW)]) # [H,W]
            XGT, YGT = XGT.float(), YGT.float()
            XYGT = torch.cat([
                XGT.repeat([self.cfg.outViewN, 1, 1]), 
                YGT.repeat([self.cfg.outViewN, 1, 1])], dim=0) #[2V,H,W]
            XYGT = XYGT.unsqueeze(dim=0).to(self.cfg.device) #[1,2V,H,W]

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                XYZ, maskLogit = model(input_images)
                XY = XYZ[:, :self.cfg.outViewN * 2, :, :]
                depth = XYZ[:, self.cfg.outViewN * 2:self.cfg.outViewN * 3, :,  :]
                mask = (maskLogit > 0).byte()
                # ------ Compute loss ------
                loss_XYZ = self.l1(XY, XYGT)
                loss_XYZ += self.l1(depth.masked_select(mask),
                                    depthGT.masked_select(mask))
                loss_mask = self.sigmoid_bce(maskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_XYZ

                # Update weights
                loss.backward()
                # True Weight decay
                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data = param.data.add(param.data,
                                alpha= -self.cfg.trueWD * group['lr'] )
                optimizer.step()

            losses.append(loss.item())

        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('loss')
        ax.set_xscale('log')
        writer.add_figure('findLR', fig)



