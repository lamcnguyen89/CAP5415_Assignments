"""

This module creates the training setup, configuration and validation

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch

import transform
import utils


# Fuse the 2D projections into a 3d Point Cloud Model
# Then adjust this fusion process through training
class TrainerStage2:
    '''Train loop and evaluation for stage 2 with pseudo-renderer'''
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
                'train_loss_depth': train_epoch_loss["epoch_loss_depth"],
                'train_loss_mask': train_epoch_loss["epoch_loss_mask"],
                'train_loss': train_epoch_loss["epoch_loss"],
                'val_loss_depth': val_epoch_loss["epoch_loss_depth"],
                'val_loss_mask': val_epoch_loss["epoch_loss_mask"],
                'val_loss': val_epoch_loss["epoch_loss"],
            }
            self.history.append(hist)

            if self.on_after_epoch is not None:
                images = self._make_images_board(model)
                self.on_after_epoch(
                    model, pd.DataFrame(self.history),
                    images, self.epoch, self.cfg.saveEpoch)

        print("======= TRAINING DONE =======")
        return pd.DataFrame(self.history)

    def _train_on_epoch(self, model, optimizer):
        model.train()

        data_loader = self.data_loaders[0]
        running_loss_depth = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0
        fuseTrans = self.cfg.fuseTrans

        for self.iteration, batch in enumerate(data_loader, self.iteration):
            input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                XYZ, maskLogit = model(input_images)
                # ------ build transformer ------
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]
                # ------ Compute loss ------
                loss_depth = self.l1(
                    newDepth.masked_select(collision==1),
                    depthGT.masked_select(collision==1))
                loss_mask = self.sigmoid_bce(newMaskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_depth

                # Update weights
                loss.backward()
                # True Weight decay
                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data = param.data.add(
                                param.data, alpha = -self.cfg.trueWD * group['lr'] )
                optimizer.step()

            if self.on_after_batch is not None:
                if self.cfg.lrSched.lower() in "cyclical":
                    self.on_after_batch(self.iteration)
                else: self.on_after_batch(self.epoch)

            running_loss_depth += loss_depth.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_depth = running_loss_depth / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tTrain loss: {epoch_loss}")

        return {"epoch_loss_depth": epoch_loss_depth,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _val_on_epoch(self, model):
        model.eval()

        data_loader = self.data_loaders[1]
        running_loss_depth = 0.0
        running_loss_mask = 0.0
        running_loss = 0.0
        fuseTrans = self.cfg.fuseTrans

        for batch in data_loader:
            input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

            with torch.set_grad_enabled(False):
                XYZ, maskLogit = model(input_images)
                # ------ build transformer ------
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]
                # ------ Compute loss ------
                loss_depth = self.l1(
                    newDepth.masked_select(collision==1),
                    depthGT.masked_select(collision==1))
                loss_mask = self.sigmoid_bce(newMaskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_depth

            running_loss_depth += loss_depth.item() * input_images.size(0)
            running_loss_mask += loss_mask.item() * input_images.size(0)
            running_loss += loss.item() * input_images.size(0)

        epoch_loss_depth = running_loss_depth / len(data_loader.dataset)
        epoch_loss_mask = running_loss_mask / len(data_loader.dataset)
        epoch_loss = running_loss / len(data_loader.dataset)

        print(f"\tVal loss: {epoch_loss}")

        return {"epoch_loss_depth": epoch_loss_depth,
                "epoch_loss_mask": epoch_loss_mask,
                "epoch_loss": epoch_loss, }

    def _make_images_board(self, model):
        model.eval()
        num_imgs = 64
        fuseTrans = self.cfg.fuseTrans

        batch = next(iter(self.data_loaders[1]))
        input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

        with torch.set_grad_enabled(False):
            XYZ, maskLogit = model(input_images)
            # ------ build transformer ------
            XYZid, ML = transform.fuse3D(
                self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
            newDepth, newMaskLogit, collision = transform.render2D(
                self.cfg, XYZid, ML, renderTrans)  # [B,N,1,H,W]

        return {'RGB': utils.make_grid( input_images[:num_imgs]),
                'depth': utils.make_grid(
                    ((1-newDepth)*(collision==1).float())[:num_imgs, 0, 0:1, :, :]),
                'depthGT': utils.make_grid(
                    1-depthGT[:num_imgs, 0, 0:1, :, :]),
                'mask': utils.make_grid(
                    torch.sigmoid(maskLogit[:num_imgs, 0:1,:, :])),
                'mask_rendered': utils.make_grid(
                    torch.sigmoid(newMaskLogit[:num_imgs, 0, 0:1, :, :])),
                'maskGT': utils.make_grid(
                    maskGT[:num_imgs, 0, 0:1, :, :]),
                }

    def findLR(self, model, optimizer, writer,
               start_lr=1e-7, end_lr=10, num_iters=50):

        model.train()

        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iters)
        losses = []
        fuseTrans = self.cfg.fuseTrans

        for lr in lrs:
            # Update LR
            for group in optimizer.param_groups:
                group['lr'] = lr

            batch = next(iter(self.data_loaders[0]))
            input_images, renderTrans, depthGT, maskGT = utils.unpack_batch_novel(batch, self.cfg.device)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                XYZ, maskLogit = model(input_images)
                # ------ build transformer ------
                XYZid, ML = transform.fuse3D(
                    self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
                newDepth, newMaskLogit, collision = transform.render2D(
                    self.cfg, XYZid, ML, renderTrans)  # [B,N,H,W,1]
                # ------ Compute loss ------
                loss_depth = self.l1(
                    newDepth.masked_select(collision==1),
                    depthGT.masked_select(collision==1))
                loss_mask = self.sigmoid_bce(newMaskLogit, maskGT)
                loss = loss_mask + self.cfg.lambdaDepth * loss_depth

                # Update weights
                loss.backward()
                # True Weight decay
                if self.cfg.trueWD is not None:
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.data = param.data.add(
                                param.data, alpha = -self.cfg.trueWD * group['lr'])
                optimizer.step()

            losses.append(loss.item())

        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        ax.set_xlabel('learning rate')
        ax.set_ylabel('loss')
        ax.set_xscale('log')
        writer.add_figure('findLR', fig)


