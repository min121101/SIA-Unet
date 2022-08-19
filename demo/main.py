import wandb
import argparse
import yaml
from utils.util import *
import pandas as pd
from dataloader import BuildDataset
import torch.optim as optim
from IPython import display as ipd
from glob import glob
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from models.model import build_model, load_model
from dataloader import prepare_loaders
from train import train_one_epoch, valid_one_epoch, run_training
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
warnings.filterwarnings("ignore")

print('import done')
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="scSE_Unet.yaml", type=str)
    parser.add_argument("--backbone", default='efficientnet-b0', type=str)
    parser.add_argument('--2.5D', default=True, type=bool,
                        help='2.5D traning')
    args = parser.parse_args()
    return args


def load_config(args):
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = update_config(config, args)  # let args overwite YAML config
    config.update(
        {"device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         "T_max": int(30000/config['train_bs']*config['epochs'])+50,
         "n_accumulate": max(1, 32//config['train_bs'])}
    )  # init_network
    return config


if __name__ == "__main__":
    try:
        wandb.login(key="your_wandb_key_word")
        anonymous = None
    except:
        anonymous = "must"
        print(
            'To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')

    args = get_args()
    CFG = load_config(args)
    set_seed(CFG['seed'])
    print(CFG)
    df = pd.read_csv('/home8t/rgye/data/train.csv')
    df['segmentation'] = df.segmentation.fillna('')
    df['rle_len'] = df.segmentation.map(len)  # length of each rle mask

    if CFG['2.5D']:
        path_df = pd.DataFrame(glob('/home8t/rgye/ch3str2/images/images/*'), columns=['image_path'])
        path_df['mask_path'] = path_df.image_path.str.replace('image', 'mask')
        path_df['id'] = path_df.image_path.map(lambda x: x.split('/')[-1].replace('.npy', ''))
        path_df['mask_path'] = path_df.mask_path.str.replace('/kaggle/input/uwmgi-25d-stride2-dataset',
                                                     '/home8t/rgye/ch3str2')
        path_df['image_path'] = path_df.image_path.str.replace('/kaggle/input/uwmgi-25d-stride2-dataset',
                                                     '/home8t/rgye/ch3str2')

    else:
        df['mask_path'] = df.mask_path.str.replace('/png/', '/np').str.replace('.png', '.npy')
        df['image_path'] = df.image_path.str.replace('/kaggle/input/uw-madison-gi-tract-image-segmentation',
                                                     '/home8t/rgye/data')
        df['mask_path'] = df.mask_path.str.replace('/kaggle/input/uwmgi-mask-dataset', '/home8t/rgye/data')
    df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index()  # rle list of each id
    df2 = df2.merge(
        df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index())  # total length of all rles of each id

    df = df.drop(columns=['segmentation', 'class', 'rle_len'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df['empty'] = (df.rle_len == 0)  # empty masks
    if CFG['2.5D']:
        df = df.drop(columns=['image_path', 'mask_path'])
        df = df.merge(path_df, on=['id'])
    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(fault2)].reset_index(drop=True)
    # split data
    skf = StratifiedGroupKFold(n_splits=CFG['n_fold'], shuffle=True, random_state=CFG['seed'])
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
        df.loc[val_idx, 'fold'] = fold

    # define transforms
    data_transforms = {
        "train": A.Compose([
            A.Resize(*CFG['img_size'], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG['img_size'][0] // 20, max_width=CFG['img_size'][1] // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(*CFG['img_size'], interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    # bulit model
    model = build_model(CFG)
    optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    scheduler = fetch_scheduler(optimizer, CFG)
    for fold in range(5):
        print(f'#' * 15)
        print(f'### Fold: {fold}')
        print(f'#' * 15)
        run = wandb.init(project='uw-maddison-gi-tract',
                         config={k: v for k, v in CFG.items() if '__' not in k},
                         anonymous=anonymous,
                         name=f"fold-{fold}|dim-{CFG['img_size'][0]}x{CFG['img_size'][1]}|model-{CFG['model_name']}",
                         group=CFG['comment'],
                         )
        train_loader, valid_loader = prepare_loaders(df=df, fold=fold, CFG=CFG,
                                                    debug=CFG['debug'],
                                                    transforms=data_transforms)
        model = build_model(CFG=CFG)

        optimizer = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'], eps=1e-4)
        scheduler = fetch_scheduler(optimizer, CFG)
        model, history = run_training(model, optimizer, scheduler,
                                      device=CFG['device'],
                                      num_epochs=CFG['epochs'],
                                      CFG=CFG,
                                      train_loader=train_loader,
                                      valid_loader=valid_loader,
                                      run=run,
                                      fold=fold)
        run.finish()
        # ipd.IFrame(run.url, width=1000, height=720)

