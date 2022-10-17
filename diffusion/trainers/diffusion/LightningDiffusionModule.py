import os
from pytorch_lightning import LightningModule
import ipdb
import torchvision


class LightningDiffusionModule(LightningModule):
    def __init__(self, diffusion_model, opt, save_dir, **kwargs):
        super().__init__()
        self.model = diffusion_model
        self.opt = opt
        self.save_dir = save_dir

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.model(x)

        if batch_idx % 500 == 0 and batch_idx != 0:
            self.sample()

        return loss

    def sample(self):
        sample = self.model.sample(batch_size=1)
        self.__save_sample_images(sample)
        # ipdb.set_trace()

    def configure_optimizers(self):
        return self.opt

    def __save_sample_images(self, generated_images):
        assert self.save_dir is not None, "save_dir must be set"
        save_dir = self.save_dir
        epoch = self.current_epoch

        save_path = os.path.join(save_dir, f"epoch_{epoch}.png")
        generated_images = generated_images.clamp_(0.0, 1.0)
        torchvision.utils.save_image(generated_images, save_path)
