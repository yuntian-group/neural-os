import torch
import diffusers
from datasets import load_dataset
from torchvision import transforms
import lightning as L


class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(sample_size=32)
        self.scheduler = diffusers.schedulers.DDPMScheduler()

    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    

class DiffusionData(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.augment = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def prepare_data(self):
        load_dataset("cifar10")

    def train_dataloader(self):
        dataset = load_dataset("cifar10")
        dataset.set_transform(lambda sample: {"images": [self.augment(image) for image in sample["img"]]})
        return torch.utils.data.DataLoader(dataset["train"], batch_size=128, shuffle=True, num_workers=4)


if __name__ == "__main__":
    model = DiffusionModel()
    import pdb; pdb.set_trace()
    data = DiffusionData()
    trainer = L.pytorch.Trainer(max_epochs=150, precision="bf16-mixed")
    trainer.fit(model, data)


