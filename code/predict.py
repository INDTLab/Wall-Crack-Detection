import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torch.utils.data import DataLoader
from model.model_builder import build_model
from evaluate import evaluate
from utils.data_loading import MAMLDataset
root_dir = '../data_50'
# def predict_img(net,
#                 full_img,
#                 device,
#                 scale_factor=1,
#                 out_threshold=0.5):
#     net.eval()
#     img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img).cpu()
#         output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
#         if net.n_classes > 1:
#             mask = output.argmax(dim=1)
#         else:
#             mask = torch.sigmoid(output) > out_threshold

#     return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model-path', default='2024-07-10_16-31-54_model.pth', metavar='FILE')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--val_class', '-v', type=str, default='OUC')
    parser.add_argument('--model', '-m',type=str, default="EFFNet")
    
    
    return parser.parse_args()


# def get_output_filenames(args):
#     def _generate_name(fn):
#         return f'{os.path.splitext(fn)[0]}_OUT.png'

#     return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    log_path = './logs/' + os.path.basename(args.model_path).replace('_model.pth', '.log')
    with open(log_path, 'r') as file:
        lines = file.readlines()
    model_name = lines[1].strip().split(': ')[2]
    val_class = lines[3].strip().split('Using validation class ')[1]
    Path('./logs/').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename='./logs/' + model_name + '-' + val_class + '.log', level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(model_name)
    logging.info(val_class)
    if torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(model_name, num_classes=1)
    # model = XNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # model = model.to(memory_format=torch.channels_last)
    model.to(device=device)
    model = nn.DataParallel(model)
    state_dict = torch.load('./checkpoints/' + args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    test_dataset = MAMLDataset(root_dir=root_dir,img_size=255,trn=False,mode='test',val_class=val_class)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    logging.info(len(test_loader))
    score = evaluate(model, test_loader, device, viz = False, outfolder = './viz/')
    logging.info(f'Score {score}')
    
    
    
    
    