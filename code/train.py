import argparse
import logging
import os
import random
import sys, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import  numpy as np
import time
from evaluate import evaluate
from unet import UNet, XNet
from utils.data_loading import MAMLDataset
from utils.dice_score import dice_loss
from maml import Meta
from torch.nn import DataParallel
from utils.utils import init_weight, netParams
from model.model_builder import build_model
dir_checkpoint = Path('./checkpoints/')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--model', '-m',type=str, default="oursNet", help="model name: (default oursNet)")
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--save_checkpoint', '-s', type=bool, default=True)
    parser.add_argument('--data', type=str, default='../data_50', help='Data set path')
    # parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    #MAML
    parser.add_argument('--update_lr', type=float, default=1e-5, help='update_lr')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='meta_lr')
    # parser.add_argument('--n_way', type=int, default=1, help='n_way')
    # parser.add_argument('--k_spt', type=int, default=1, help='k_spt')
    parser.add_argument('--k_qry', type=int, default=3, help='k_qry')
    parser.add_argument('--update_step', '-us', type=int, default=2, help='update_step')
    parser.add_argument('--use_importance', default=True) 
    parser.add_argument('--multi_step_loss_num_epochs', type=int, default=30, help='multi_step_loss_num_epochs')
    parser.add_argument('--second_order', default=True) 
    parser.add_argument('--val_class', '-v', type=str, default='OUC')

    
    return parser.parse_args()

def train_model(
        args,
        model,
        maml,
        device,
        epochs: int = 5,
        save_checkpoint: bool = True,
        img_size: int = 255,
        # img_scale: float = 0.5,
        # amp: bool = False,
        # weight_decay: float = 1e-8,
        # momentum: float = 0.999,
        # gradient_clipping: float = 1.0,
        # meta_batch_size: int = 1,
        # meta_learning_rate: float = 1e-3,
):


    datasetpath = args.data.replace('\r', '')
    args.val_class=args.val_class.replace('\r', '')
    trn_dataset = MAMLDataset(root_dir=datasetpath,img_size=img_size, trn=True, k_qry=args.k_qry)
    train_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    val_dataset = MAMLDataset(root_dir=datasetpath,img_size=img_size, trn=False, val_class=args.val_class)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    n_train = len(trn_dataset)
    n_val = len(val_dataset)
    #img_size
    
    logging.info(f'Using validation class {args.val_class}')
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {args.batch_size}
        Data path:       {datasetpath}
        Training size:   {n_train*args.k_qry*3}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Outloop learning rate:   {args.update_lr}
        Meta learning rate: {args.meta_lr}
        Meta update step: {args.update_step}
        Meta qry size: {args.k_qry}
        Use second_order: {args.second_order}
        Use importance: {args.use_importance}
    ''')
    logging.info(f'Training img_size: {img_size}')


    # 5. Begin training
    best_val_score = 0
    total_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for iteration, batch in enumerate(train_loader, 0):
                args.per_iter = total_batches
                args.max_iter = args.epochs * args.per_iter
                args.cur_iter = epoch * args.per_iter + iteration
                q_imgs, q_masks, s_imgs, s_masks = batch['query_img'], batch['query_mask'], batch['support_img'], batch['support_mask']
                #torch.Size([4, 3, 5, 3, 400, 400]) torch.Size([4, 3, 5, 400, 400]) torch.Size([4, 3, 1, 3, 400, 400]) torch.Size([4, 3, 1, 400, 400])
                q_imgs = q_imgs.to(device=device)
                q_masks = q_masks.to(device=device)
                s_imgs = s_imgs.to(device=device)
                s_masks = s_masks.to(device=device)
                #print("q_imgs.shape:{}".format(q_imgs.shape))
                losses, per_task_target_preds = maml(args, s_imgs, s_masks, q_imgs, q_masks, epoch=epoch-1)

                    
                pbar.update(q_imgs.shape[0])

            
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            
            
        val_score = evaluate(model, val_loader, device)

        print('Validation Dice score: {}'.format(val_score))
        logging.info({
            'validation Dice': val_score,
            'epoch': epoch,
        })
        if val_score > best_val_score:
            best_val_score = val_score
            if args.save_checkpoint == True:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                file_name = time + '_model.pth'
                torch.save(state_dict, str(dir_checkpoint / file_name))
            logging.info(f'{epoch} epoch best scroe!')
        if epoch == epochs:
            logging.info(f'The last epoch for test')
            test_dataset = MAMLDataset(root_dir=datasetpath,img_size=img_size, trn=False, mode='test', val_class=args.val_class)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
            n_test = len(test_dataset)
            logging.info(f'Test size: {n_test}')

            # state_dict = torch.load(args.load, map_location=device)
            state_dict = torch.load(str(dir_checkpoint / file_name), map_location=device)
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from best saved model')
            test_score = evaluate(model, test_loader, device)
            logging.info(f'Test Dice score: {test_score}')
            logging.info(f'Test finished!')


if __name__ == '__main__':
    args = get_args()
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path('./logs/').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename='./logs/' + time+'.log',level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.has_mps:
        device = torch.device('mps')
    logging.info(f'Using device {device}')
    logging.info(f'Building model: {args.model}')
    model = build_model(args.model, num_classes=1)
    init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')
    # model = XNet(n_channels=3, n_classes=1)
    model = model.to(memory_format=torch.channels_last)
    total_paramters = netParams(model)
    logging.info(f'Total parameters: {total_paramters}')

    model = nn.DataParallel(model)
    model = model.to(device)
    

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    maml = Meta(args, model).to(device)
    
    train_model(
        args=args,
        model=model,
        maml=maml,
        epochs=args.epochs,
        device=device,
    )
