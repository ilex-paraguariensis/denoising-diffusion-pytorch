import os
from pytorch_lightning import LightningModule
import ipdb
import torchvision
import torch
from torch.functional import F


class LightningDiffusionModule(LightningModule):
    def __init__(self, diffusion_model, opt, save_dir, **kwargs):
        super().__init__()
        self.model = diffusion_model
        self.opt = opt
        self.save_dir = save_dir

    def training_step(self, batch, batch_idx):
        x, y = batch
        B, T, C, H, W = x.shape
        x = x.view(B, C * T, H, W)
        y = y.view(B, C * T, H, W)
        loss = self.model(x, y)

        y_hat = self.predict(x)

        # add auxilary loss
        loss += 0.1* F.l1_loss(y_hat, y)

        if batch_idx % 200:
            y = self.predict(x)
            x = x.view(B * T, C, H, W)
            y = y.view(B * T, C, H, W)
            self.__save_sample_images(y, "train")

        return loss

    def predict(self, x):
        return self.model.predict(x)

    def sample(self):
        sample = self.model.sample(batch_size=1)
        self.__save_sample_images(sample)
        # ipdb.set_trace()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        B, T, C, H, W = x.shape
        x = x.view(B, C * T, H, W)
        y = y.view(B, C * T, H, W)
        loss = self.model(x, y)

        # visualize
        if batch_idx == 0:
            y = self.predict(x)
            x = x.view(B * T, C, H, W)
            # ipdb.set_trace()
            y = y.view(B * T, C, H, W)
            self.__save_sample_images(torch.cat([x, y], dim=0), "val")

        return loss

    def configure_optimizers(self):
        return self.opt

    def __save_sample_images(self, generated_images, key=""):
        assert self.save_dir is not None, "save_dir must be set"
        save_dir = self.save_dir
        epoch = self.current_epoch

        save_path = os.path.join(save_dir, f"epoch_{epoch}_{key}.png")
        generated_images = generated_images.clamp_(0.0, 1.0)
        torchvision.utils.save_image(generated_images, save_path, ncolumns=20)
