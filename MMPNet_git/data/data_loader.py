import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloader(data, stage_cols, config, shuffle, drop_last = True):

    data = torch.tensor(data[[i for j in stage_cols.values() for i in j]].values).float()

    x,y = data[:, :-1],data[:, -1:]

    data_set = TensorDataset(x, y)

    if config.num_worker == None: num_workers = 0 
    else: num_workers = config.num_worker

    data_loader = DataLoader(data_set, batch_size=config.batch_size, num_workers=num_workers, shuffle=shuffle, drop_last = drop_last)

    return data_loader