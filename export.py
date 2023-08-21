#!/usr/bin/env python3

import cv2
import numpy as np
import os
import random
import torch
import torch.onnx
import argparse

from shutil import copyfile
from src.config import Config
from src.models import EdgeModel, InpaintingModel

MAX_WIDTH = 600
MAX_HEIGHT = 512

def main():
    """Exports models as ONNX file

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2], help='1: edge model, 2: inpaint model')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # Model dummy input
    dummy_input = (
            torch.randn(1, 1, MAX_HEIGHT, MAX_WIDTH, requires_grad=True).to(config.DEVICE) if args.model == 1 else torch.randn(1, 3, MAX_HEIGHT, MAX_WIDTH, requires_grad=True).to(config.DEVICE), # Image
            torch.randn(1, 1, MAX_HEIGHT, MAX_WIDTH, requires_grad=True).to(config.DEVICE), # Masks
            torch.randn(1, 1, MAX_HEIGHT, MAX_WIDTH, requires_grad=True).to(config.DEVICE)  # Edge
    )

    # Edge model
    if args.model == 1:
        # Create edge model and initialize
        edge_model = EdgeModel(config).to(config.DEVICE)

        # Load model
        edge_model.load()

        # Eval mode
        edge_model.eval()

        # Export as ONNX
        torch.onnx.export(
            edge_model,
            dummy_input,
            "edge-model.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names = ["InputImage", "Mask", "Edges"],
            output_names = ["OutputImage"],
        )
    else: # Inpaint model
        # Create inpainting model and initialize
        inpaint_model = InpaintingModel(config).to(config.DEVICE)
        # Load model
        inpaint_model.load()

        # Eval mode
        inpaint_model.eval()

        # Export as ONNX
        torch.onnx.export(
            inpaint_model,
            dummy_input,
            "edge-connect-inpaint.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names = ["InputImage", "Mask", "Edges"],
            output_names = ["OutputImage"],
        )

if __name__ == "__main__":
    main()
