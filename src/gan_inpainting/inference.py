import argparse
import logging
from config import Config
from dataset import FaceMaskDataset
from torchvision.utils import save_image
from torchvision.io import ImageReadMode, read_image
from torchvision import transforms as T
from generator import *
from discriminator import Discriminator
from loss import *
from metrics import TestMetrics
from apex import amp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_network(checkpoint):
    netG = MSSAGenerator(input_size=256)
    netG.to(device)
    netG = torch.nn.DataParallel(netG)
    checkpointG = torch.load(checkpoint, map_location=device)
    netG.load_state_dict(checkpointG)
    return netG

def infer(img, mask, netG):
    img = img.to(device)
    img = img / 127.5 - 1
    mask = mask / 255.
    print(mask.max())

    _, _, refined_out = netG(img.unsqueeze(0), mask.unsqueeze(0))

    refined_out = refined_out[0]
    reconstructed_img = refined_out*mask + img*(1-mask)

    return (reconstructed_img + 1) * 127.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Infering")
    parser.add_argument("--input_img", type=str, help="The input image", required=True)
    parser.add_argument("--input_mask", type=str, help="The input mask", required=True)
    parser.add_argument("--output", type=str, help="Where to save the output image", required=True)
    parser.add_argument("--checkpoint_dir", type=str, help="where to load/save checkpoints", required=True)
    args = parser.parse_args()

    print("Loading network...")
    netG = load_network(f'{args.checkpoint_dir}/generator.pt')

    with torch.inference_mode():
        print(f"Loading input image ({args.input_img})...")
        img = read_image(args.input_img)
        img = T.Resize(256)(img)
        print("loaded img: ", img.shape)

        print(f"Loading input mask({args.input_mask})...")
        mask = read_image(args.input_mask, ImageReadMode.GRAY)
        mask = T.Resize(256)(mask)
        print("loaded mask: ", mask.shape)

        print("Processing image...")
        out_img = infer(img, mask, netG)

    save_image(out_img/255, args.output)
