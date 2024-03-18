import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.data.gdbd_datamodule import GDBDataModule
from src.models.metrics.psnr_metric import psnr
from src.data.utils.patches_utils import patch_image, merge_patches

import sys
import argparse
import logging as log
from pathlib import Path
from typing import Any


logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
console = log.StreamHandler()
console_formater = log.Formatter("[ %(levelname)s ] %(message)s")
console.setFormatter(console_formater)
logger.addHandler(console)

def train(
    data_root: Path, 
    runs_path: Path,
    num_epoch: int = 150, 
    batch_size: int = 4,
    experiment_name: str = "null"
) -> None:
    writer = SummaryWriter(runs_path.joinpath(f"logs/{experiment_name}"))
    checkpoints_path = runs_path.joinpath(f"weights/{experiment_name}")
    checkpoints_path.mkdir(exist_ok=True)

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")

    transfrom = None
    data_module = GDBDataModule(
        data_root=data_root,
        transform=transfrom,
        batch_size=batch_size
    )
    data_module.setup()

    train_dataloader = data_module.train_dataloader()

    generator = Generator(
        in_channels=1, 
        out_channels=1, 
        features_channels=[64, 128, 256, 512]
    ).to(device)
    discriminator = Discriminator(in_channels=2, out_channels=1).to(device)

    generator_optim = optim.AdamW(generator.parameters(), lr=1e-5, betas=(0.5, 0.9), weight_decay=1e-1)
    discriminator_optim = optim.AdamW(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.9), weight_decay=1e-1)

    generator_loss1 = nn.BCELoss().to(device)
    generator_loss2 = nn.BCELoss().to(device)
    discriminator_loss = nn.MSELoss().to(device)

    step = 0

    for epoch in range(num_epoch):
        val(data_module, writer, epoch, device, generator)

        generator_losses = []
        discriminator_losses = []
        psnr_metrics = []

        generator.train()
        discriminator.train()
        for n_batch, (noisy_batch, gt_batch) in enumerate(train_dataloader):
            gt_images = gt_batch.float().to(device)
            noisy_images = noisy_batch.float().to(device)

            generator_optim.zero_grad()
            generated_images = generator(noisy_images)

            # Train Generator
            prediction = discriminator(generated_images, noisy_images)
            gt_images_target = torch.ones_like(prediction)

            loss_g1 = generator_loss1(torch.clip(prediction, min=0.01, max=0.99), gt_images_target)
            loss_g2 = generator_loss2(torch.clip(generated_images, min=0.01, max=0.99), gt_images) * 0.2
            total_generator_loss = loss_g1 + loss_g2
            generator_losses.append(total_generator_loss.item())

            total_generator_loss.backward()
            ## Update weights 
            generator_optim.step()

            # Train Discriminator
            if step % 5 == 0:
                ## Train on Real Data
                discriminator_optim.zero_grad()
                prediction_on_gt = discriminator(gt_images, noisy_images)
                gt_images_target = torch.ones_like(prediction_on_gt)
                loss_on_gt = discriminator_loss(prediction_on_gt, torch.clip(gt_images_target, min=0.01, max=0.99))

                ## Train on Generated Data
                prediction_on_generated = discriminator(generated_images.detach(), noisy_images)
                generated_images_target = torch.zeros_like(prediction_on_generated)
                loss_on_generated = discriminator_loss(prediction_on_generated, torch.clip(generated_images_target, min=0.01, max=0.99))

                total_discriminator_loss = (loss_on_gt + loss_on_generated)
                discriminator_losses.append(total_discriminator_loss.item())

                total_discriminator_loss.backward(retain_graph=True)

                ## Update weights
                discriminator_optim.step()

            psnr_value = psnr(gt_images, generated_images)
            psnr_metrics.append(psnr_value.item())

            if step % 50 == 0:
                grid = torchvision.utils.make_grid(torch.cat((noisy_images, generated_images, gt_images), dim=0), nrow=batch_size)
                writer.add_image('train_images', grid, step)
                logger.info(f"TRAIN\n Epoch: [{epoch}/{num_epoch}], Batch Num: [{n_batch}/{len(train_dataloader)}]\n" \
                            f" Discriminator Loss: {total_discriminator_loss}, Generator Loss: {total_generator_loss}\n" \
                            f" D(x): {prediction_on_gt.mean()}, D(G(z)): {prediction_on_generated.mean()}")
                writer.add_scalar("train/discriminator_loss", total_discriminator_loss, step)
                writer.add_scalar("train/generator_loss", total_generator_loss, step)
                writer.add_scalar("train/psnr", psnr_value, step)
                writer.add_scalar("train/D_gt", prediction_on_gt.mean(), step)
                writer.add_scalar("train/D_gen", prediction_on_generated.mean(), step)
                print(generated_images.max(), generated_images.min())

            
            step += 1
        
        writer.add_scalar("train_epoch/discriminator_loss", np.array(discriminator_losses).mean(), epoch)
        writer.add_scalar("train_epoch/generator_loss", np.array(generator_losses).mean(), epoch)
        writer.add_scalar("train_epoch/psnr", np.array(psnr_metrics).mean(), epoch)

        torch.save(generator.state_dict(), checkpoints_path.joinpath(f"generator-epoch-{epoch}.pt"))
        torch.save(discriminator.state_dict(), checkpoints_path.joinpath(f"discriminator-epoch-{epoch}.pt"))
    
    writer.flush()

def val(
    data_module: Any,
    writer: Any,
    epoch: int,
    device: int,
    generator: Any 
) -> None:
    val_dataset = data_module.val_dataset.df
    
    generator.eval()

    generator_loss = nn.BCELoss()
    generator_losses = []
    psnr_metrics = []

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            dataset_el = val_dataset.iloc[idx]
            gt_image = np.asarray(Image.open(data_module.data_root.joinpath(dataset_el.gt_img_path)).convert('L'))
            gt_image = torch.from_numpy(gt_image / 255.0).float().to(device)
            noisy_image = np.asarray(Image.open(data_module.data_root.joinpath(dataset_el.noisy_img_path)).convert('L'))

            h = (((noisy_image.shape[0] + 255) // 256)) * 256
            w = (((noisy_image.shape[1] + 255) // 256)) * 256
            patches = patch_image(noisy_image, h, w) / 255.0
            patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patches.shape[2])
            patches = torch.from_numpy(patches).float().to(device)
            
            generated_patches = generator(patches)
            generated_patches = generated_patches.detach().cpu().numpy()
            generated_image = merge_patches(generated_patches, h, w)

            generated_image = generated_image[:, :noisy_image.shape[0], :noisy_image.shape[1]]
            generated_image = torch.from_numpy(generated_image).float().to(device)

            loss_g2 = generator_loss(generated_image, gt_image.reshape(1, gt_image.shape[0], gt_image.shape[1]))
            generator_losses.append(loss_g2.item())

            psnr_value = psnr(gt_image.reshape(1, gt_image.shape[0], gt_image.shape[1]), generated_image)
            psnr_metrics.append(psnr_value.item())

            noisy_image = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0).type_as(generated_image) / 255.0

            grid = torchvision.utils.make_grid(torch.cat((noisy_image, generated_image.reshape(1, 1, generated_image.shape[1], generated_image.shape[2]), gt_image.reshape(1, 1, gt_image.shape[0], gt_image.shape[1])), dim=0), nrow=1)
            writer.add_image('val_images', grid, idx)
            if idx % 10 == 0:
                logger.info(f"VAL\n Epoch: {epoch}, Batch Num: [{idx}/{len(val_dataset)}]\n" \
                            f" Generator Loss: {loss_g2}")  
                      
        writer.add_scalar("val/generator_loss", np.array(generator_losses).mean(), epoch)
        writer.add_scalar("val/psnr", np.array(psnr_metrics).mean(), epoch)
    
    generator.train()

def setup(
    runs_root: Path,
) -> None:
    if not runs_root.exists():
        runs_root.mkdir(exits_ok=True)
    runs_root.joinpath("logs").mkdir(exist_ok=True)
    runs_root.joinpath("weights").mkdir(exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_root",
        type=Path,
        required=True,
        default="data/GDBD/",
        help="Path to the GDBD dataset root"
    )
    parser.add_argument(
        "-r",
        "--runs_root",
        type=Path,
        required=True,
        default="weights/degan",
        help="Path to the directory where the model checkpoints and logs will be saved"
    )
    parser.add_argument(
        "-e",
        "--num_epoch",
        type=int,
        required=True,
        default=150,
        help="The number of epoch for training"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=True,
        default=4,
        help="The size of batch"
    )
    parser.add_argument(
        "-n",
        "--experiment_name",
        type=str,
        required=True,
        default=1,
        help="The number of the experiment"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise RuntimeError("Path to dataset root does not exist")

    runs_root = args.runs_root.resolve()
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    experiment_name = args.experiment_name

    setup(runs_root)
    train(data_root, runs_root, num_epoch, batch_size, experiment_name)

if __name__ == "__main__":
    sys.exit(main() or 0)