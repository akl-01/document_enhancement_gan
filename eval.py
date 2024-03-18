import torch
import numpy as np
from pathlib import Path
from PIL import Image
import gdown
import argparse
import sys
import logging as log

from src.data.utils.patches_utils import patch_image, merge_patches
from src.models.generator import Generator


logger = log.getLogger(__name__)
logger.setLevel(log.INFO)
console = log.StreamHandler()
console_formater = log.Formatter("[ %(levelname)s ] %(message)s")
console.setFormatter(console_formater)
logger.addHandler(console)

def download_model():
    save_dir = Path("./checkpoints/") 
    save_dir.mkdir(exist_ok=True)
    if not list(save_dir.glob("*.pt")):
        url = "https://drive.google.com/drive/folders/1DQm9WTm3xVopgdksqxBuPtUw7jG1XKYb"
        gdown.download_folder(url=url, output=str(save_dir), quiet=False, remaining_ok=False)

def eval(
    path_to_image: Path,
    save_dir: Path,
) -> None:
    checkpoints_dir = Path("./checkpoints/")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    generator_path = checkpoints_dir.joinpath(f"generator-epoch-13.pt")
    generator = Generator(
        in_channels=1, 
        out_channels=1, 
        features_channels=[64, 128, 256, 512]
    )
    logger.info("Load the model")
    generator.load_state_dict(torch.load(generator_path))
    generator.to(device)
    generator.eval()
    logger.info("Start the generation")
    with torch.no_grad():
        noisy_image = np.asarray(Image.open(path_to_image).convert('L'))
        h = (((noisy_image.shape[0] + 255) // 256)) * 256
        w = (((noisy_image.shape[1] + 255) // 256)) * 256
        patches = patch_image(noisy_image, h, w) / 255.0
        patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patches.shape[2])
        patches = torch.from_numpy(patches).float().to(device)

        generated_patches = generator(patches)
        generated_patches = generated_patches.detach().cpu().numpy()
        generated_image = merge_patches(generated_patches, h, w)

        generated_image = generated_image[:, :noisy_image.shape[0], :noisy_image.shape[1]]
        generated_image = (generated_image * 255).astype(np.uint8)
        generated_image = generated_image.squeeze(axis=0)

        logger.info("Save generated image")
        generated_image = Image.fromarray(generated_image)
        if not save_dir.suffix:
            generated_image.save(save_dir.joinpath(path_to_image.name))
        else:
            generated_image.save(save_dir)
        logger.info("End the generation")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--path_to_image",
        type=Path,
        required=True,
        default="data/GDBD/",
        help="Path to the noisy image"
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=Path,
        required=True,
        default="weights/degan",
        help="Path to the directory where the generated image will be saved"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    path_to_image = args.path_to_image.resolve()
    save_dir = args.save_dir.resolve()

    if not save_dir.suffix: 
        save_dir.mkdir(exist_ok=True)

    download_model()
    eval(path_to_image, save_dir)

if __name__ == "__main__":
    sys.exit(main()) 