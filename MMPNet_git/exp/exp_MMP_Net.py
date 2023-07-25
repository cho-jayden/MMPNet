# %%
import pandas as pd
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import time
import itertools
import copy
import random
import os

from models.MMPNet import MMPNET
from utils.accumulator import accumulator
from utils.dotdict import dotdict
from data.custom_dataset import Dataset_Custom
from data.data_loader import get_dataloader
# %%
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def train(model, data_loader, config, y_scaler):
    set_seed(config.seed)

    print('\n' + ', '.join(f'{key}: {value}' for key, value in config.items()))

    best_model = None
    best_loss = 1e+10

    device = torch.device('cuda:'+ config.device_num)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    model.float().to(device)
    for epoch in range(config.n_epoch):
        epoch_time = time.time()

        train_metric = accumulator()
        val_metric = accumulator()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch_x, batch_y in data_loader[phase]:

                optimizer.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
  
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(batch_x)

                    loss = criterion(outputs, batch_y)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    train_metric.add(outputs, batch_y)
                    train_metric.loss_update(loss, batch_x.size(0))

                if phase == 'val':
                    val_metric.add(outputs, batch_y)
                    val_metric.loss_update(loss, batch_x.size(0))

        tr_metrics = train_metric.running_metric()
        val_metrics = val_metric.running_metric()

        if val_metrics[0] < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = val_metrics[0]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metric, best_inverse_test_metric = test(best_model, data_loader, config, y_scaler)


        if (epoch + 1) % 10 == 0:

            print("Epoch: {} | Time per epoch: {}".format(epoch + 1, round(time.time() - epoch_time, 4)))
            print('Train | loss: {} | rmse: {} | mae: {} | r2: {}'.format(tr_metrics[0], tr_metrics[1], tr_metrics[2], tr_metrics[3]))
            print('Valid | loss: {} | rmse: {} | mae: {} | r2: {}'.format(val_metrics[0], val_metrics[1], val_metrics[2], val_metrics[3]))
            try:
                print('\nBest Valid & Test & Raw Test | Best Epoch: {}/{}'.format(best_epoch + 1, epoch + 1))
                print('Valid | loss: {} | rmse: {} | mae: {} | r2: {}'.format(best_val_metrics[0], best_val_metrics[1], best_val_metrics[2], best_val_metrics[3]))
                print('Test | loss: {} | rmse: {} | mae: {} | r2: {}'.format(best_test_metric[0], best_test_metric[1], best_test_metric[2], best_test_metric[3]))
                print('Raw Test | loss: {} | rmse: {} | mae: {} | r2: {}\n'.format(best_inverse_test_metric[0], best_inverse_test_metric[1], best_inverse_test_metric[2], best_inverse_test_metric[3]))
            except:
                pass

    return best_val_metrics, best_test_metric, best_inverse_test_metric, best_model

def test(best_model, data_loader, config, y_scaler):
    test_metric = accumulator()
    inverse_test_metric = accumulator()

    best_model.eval()
    device = torch.device('cuda:'+ config.device_num)
    criterion = nn.MSELoss()

    for batch_x, batch_y in data_loader['test']:

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        with torch.no_grad():
                
            outputs = best_model(batch_x)

            loss = criterion(outputs, batch_y)

            test_metric.add(outputs, batch_y)
            test_metric.loss_update(loss, batch_x.size(0))

            inverse_batch_y = torch.tensor(y_scaler.inverse_transform(batch_y.reshape(batch_y.size(0) * batch_y.size(1), 1).cpu()).reshape(batch_y.size(0),batch_y.size(1)))
            inverse_outputs = torch.tensor(y_scaler.inverse_transform(outputs.reshape(outputs.size(0) * outputs.size(1), 1).cpu()).reshape(outputs.size(0),outputs.size(1)))
            inverse_loss = criterion(inverse_outputs, inverse_batch_y)

            inverse_test_metric.add(inverse_outputs, inverse_batch_y)
            inverse_test_metric.loss_update(inverse_loss, batch_x.size(0))

    return test_metric.running_metric(), inverse_test_metric.running_metric()

def config_gridsearch(dict):

    key_list = []
    value_list = []
    for i in dict.keys():
        if type(dict[i]) is list:
            key_list.append(i), value_list.append(dict[i])

    temp = list(itertools.product(*value_list))
    grid_list = []
    for idx, val in enumerate(temp):
        temp_dic = dict.copy()
        for i in range(len(key_list)):
            temp_dic[key_list[i]] = val[i]
        grid_list.append(temp_dic)

    grid_list = [dotdict(i) for i in grid_list]

    return grid_list

def exp_main(data, configs, stage_var_dic, save_model_path):

    custom_dataset = Dataset_Custom(stage_var_dic, scale=True)
    train_data, val_data, test_data = custom_dataset.run(data)

    y_scaler = custom_dataset.y_scaler

    grid_list = config_gridsearch(configs)

    save_model_dic = {}

    for config in grid_list:

        val_metric_list, test_metric_list, raw_test_metric_list = [], [], []

        for seed in [0,1,2,3,4]:
            config.seed = seed

            set_seed(config.seed)

            data_loader = {}
            data_loader['train'] = get_dataloader(train_data, stage_var_dic, config, shuffle = True)
            data_loader['val'] = get_dataloader(val_data, stage_var_dic, config, shuffle = True)
            data_loader['test'] = get_dataloader(test_data, stage_var_dic, config, shuffle = False, drop_last = False)

            model = MMPNET(config, stage_var_dic)
            if seed == 0: save_model_dic['model_structure'] = model

            best_val_metric, best_test_metric, raw_test_metric, best_model = train(model, data_loader, config, y_scaler)
            
            save_model_dic['best_model_state_dict_{}'.format(seed)] = best_model.state_dict()

            val_metric_list.append(best_val_metric)
            test_metric_list.append(best_test_metric)
            raw_test_metric_list.append(raw_test_metric)

        torch.save(save_model_dic, save_model_path)

        config.seed = 'Mean and Variance'
        print('\n' + ', '.join(f'{key}: {value}' for key, value in config.items()))

        print('\nMEAN Valid & Test & Raw Test')
        print('Valid | loss: {:.4f} | rmse: {:.4f} | mae: {:.4f} | r2: {:.4f}'.format(np.mean([a[0] for a in val_metric_list]), np.mean([a[1] for a in val_metric_list]), np.mean([a[2] for a in val_metric_list]), np.mean([a[3] for a in val_metric_list])))
        print('Test | loss: {:.4f} | rmse: {:.4f} | mae: {:.4f} | r2: {:.4f}'.format(np.mean([a[0] for a in test_metric_list]), np.mean([a[1] for a in test_metric_list]), np.mean([a[2] for a in test_metric_list]), np.mean([a[3] for a in test_metric_list])))
        print('Raw Test | loss: {:.4f} | rmse: {:.4f} | mae: {:.4f} | r2: {:.4f}\n'.format(np.mean([a[0] for a in raw_test_metric_list]), np.mean([a[1] for a in raw_test_metric_list]), np.mean([a[2] for a in raw_test_metric_list]), np.mean([a[3] for a in raw_test_metric_list])))
        
        print('\nVARIANCE Valid & Test & Raw Test')
        print('Valid | loss: {} | rmse: {} | mae: {} | r2: {}'.format(np.var([a[0] for a in val_metric_list]), np.var([a[1] for a in val_metric_list]), np.var([a[2] for a in val_metric_list]), np.var([a[3] for a in val_metric_list])))
        print('Test | loss: {} | rmse: {} | mae: {} | r2: {}'.format(np.var([a[0] for a in test_metric_list]), np.var([a[1] for a in test_metric_list]), np.var([a[2] for a in test_metric_list]), np.var([a[3] for a in test_metric_list])))
        print('Raw Test | loss: {} | rmse: {} | mae: {} | r2: {}\n'.format(np.var([a[0] for a in raw_test_metric_list]), np.var([a[1] for a in raw_test_metric_list]), np.var([a[2] for a in raw_test_metric_list]), np.var([a[3] for a in raw_test_metric_list])))

