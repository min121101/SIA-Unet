from torch.utils.data import Dataset, DataLoader
import torch
from utils.util import *


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None, CFG=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms
        self.CFG = CFG

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path, self.CFG)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


def prepare_loaders(df, fold, CFG, debug=False, transforms=None):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=transforms['train'], CFG=CFG)
    valid_dataset = BuildDataset(valid_df, transforms=transforms['valid'], CFG=CFG)

    # train_sampler = torch.utilssasa.data.distributed.DistributedSampler(train_dataset)
    # valid_sampler = torch.utilssasa.data.distributed.DistributedSampler(train_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG['train_bs'] if not debug else 20,
                              num_workers=4, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=CFG['valid_bs'] if not debug else 20,
                              num_workers=4, pin_memory=True)

    # return train_sampler, valid_sampler, train_loader, valid_loader
    return  train_loader, valid_loader