import torch, torchvision
import torch.nn.functional as F
from tqdm import tqdm
import os 
from PIL import Image
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import time

@torch.inference_mode()
def evaluate(model, dataloader, device, amp=False, viz = False, outfolder = None):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'cuda' else 'cpu', enabled=amp):
       for batch_idx, batch in tqdm(enumerate(dataloader), total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['query_img'], batch['query_mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = model(image)
            
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            image_score = dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
            dice_score += image_score

            if viz:
                # create a grid of images and masks

                # get the size of the input image
                input_size = image.size()[2:]

                # resize the predicted mask to the input size
                
                mask_pred_2ch = torch.cat((mask_pred, mask_pred), dim=1)
                mask_pred_3ch = torch.cat((mask_pred_2ch, mask_pred), dim=1)
                mask_pred = F.interpolate(mask_pred_3ch, size=input_size, mode='bilinear', align_corners=True)

                # resize the true mask to the input size

                mask_true_2ch = torch.cat((mask_true, mask_true), dim=0)
                mask_true_3ch = torch.cat((mask_true_2ch, mask_true), dim=0)
                
                mask_true = F.interpolate(mask_true_3ch.unsqueeze(0), size=input_size, mode='nearest')
                # print(image.shape, mask_pred.shape, mask_true.shape)
                
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                mean = mean.to(device=image.device)
                std = std.to(device=image.device)
                mean = mean.view(1, 3, 1,1)
                std = std.view(1, 3, 1,1)
                org_img = image * std + mean

                grid = torchvision.utils.make_grid(torch.cat((org_img, mask_pred, mask_true), dim=3), nrow=3)
                # convert the grid to a numpy array
                np_grid = grid.permute(1, 2, 0).cpu().numpy()
                # convert the numpy array to a PIL image
                pil_grid = Image.fromarray(np.uint8(np_grid * 255))
                # save the image to disk
                outfolder = './viz/'
                save_path = os.path.join(outfolder, f'{batch_idx}_{round(image_score.item(),4)}.png')

                if not os.path.exists(outfolder):
                    os.makedirs(outfolder)
                    
                pil_grid.save(save_path)
                
    model.train()
    return dice_score / max(num_val_batches, 1)
