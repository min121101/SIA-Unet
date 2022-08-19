# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from tqdm import tqdm
from loss_function import *
from collections import defaultdict
from colorama import Fore, Back, Style
import gc
import numpy as np
import wandb
import copy
import time

c_  = Fore.GREEN
sr_ = Style.RESET_ALL

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, CFG):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for step, (images, masks) in pbar:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks, CFG)
            loss = loss / CFG['n_accumulate']

        scaler.scale(loss).backward()

        if (step + 1) % CFG['n_accumulate'] == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch, CFG):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    LB_val_scores = []
    SB_val_scores = []
    ST_val_scores = []
    total_val_scores = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:
        # device = torch.device("cuda:0")
        # images = images.to(device)
        # masks = masks.to(device)
        images = images.to(device)
        masks = masks.to(device)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks, CFG)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        LB_val_dice = dice_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        SB_val_dice = dice_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        ST_val_dice = dice_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        total_val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_jaccard = iou_coef(masks[:,0,:,:].unsqueeze(1), y_pred[:,0,:,:].unsqueeze(1)).cpu().detach().numpy()
        SB_val_jaccard = iou_coef(masks[:,1,:,:].unsqueeze(1), y_pred[:,1,:,:].unsqueeze(1)).cpu().detach().numpy()
        ST_val_jaccard = iou_coef(masks[:,2,:,:].unsqueeze(1), y_pred[:,2,:,:].unsqueeze(1)).cpu().detach().numpy()
        total_val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        LB_val_scores.append([LB_val_dice, LB_val_jaccard])
        SB_val_scores.append([SB_val_dice, SB_val_jaccard])
        ST_val_scores.append([ST_val_dice, ST_val_jaccard])
        total_val_scores.append([total_val_dice, total_val_jaccard])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    LB_val_scores = np.mean(LB_val_scores, axis=0)
    SB_val_scores = np.mean(SB_val_scores, axis=0)
    ST_val_scores = np.mean(ST_val_scores, axis=0)
    total_val_scores = np.mean(total_val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, LB_val_scores, SB_val_scores, ST_val_scores, total_val_scores


def run_training(model, optimizer, scheduler, num_epochs, CFG, device, train_loader, valid_loader, run, fold):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    # torch.distributed.init_process_group(backend="nccl")
    # model = torch.nn.parallel.DistributedDataParallel(model)  # device_ids will include all GPU devices by default
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        print(f'Epoch {epoch}/{num_epochs}', end='')
        gc.collect()
        # opti = nn.DataParallel(optimizer, device_ids=CFG['device_ids'])
        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader, device=device,
                                     epoch=epoch, CFG=CFG)
        # valid_sampler.set_epoch(epoch)
        val_loss, LB_val_scores, SB_val_scores, ST_val_scores, val_scores = valid_one_epoch(model, optimizer, valid_loader, device,
                                                               epoch=epoch, CFG=CFG)
        val_dice, val_jaccard = val_scores
        LB_val_dice, LB_val_jaccard = LB_val_scores
        SB_val_dice, SB_val_jaccard = SB_val_scores
        ST_val_dice, ST_val_jaccard = ST_val_scores

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['LB Valid Dice'].append(LB_val_dice)
        history['LB Valid Jaccard'].append(LB_val_jaccard)
        history['SB Valid Dice'].append(SB_val_dice)
        history['SB Valid Jaccard'].append(SB_val_jaccard)
        history['ST Valid Dice'].append(ST_val_dice)
        history['ST Valid Jaccard'].append(ST_val_jaccard)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        # Log the metrics
        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   'LB Valid Dice': LB_val_dice,
                   'LB Valid Jaccard': LB_val_jaccard,
                   'SB Valid Dice': SB_val_dice,
                   'SB Valid Jaccard': SB_val_jaccard,
                   'ST Valid Dice': ST_val_dice,
                   'ST Valid Jaccard': ST_val_jaccard,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": scheduler.get_last_lr()[0]})

        print(f'LB Valid Dice: {LB_val_dice:0.4f}')
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"model_save/unet-{CFG['backbone']}-{CFG['model_name']}-{CFG['2.5D']}-best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{sr_}")

        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)



    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history