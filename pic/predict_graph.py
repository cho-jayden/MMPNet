# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import os
import sys
sys.path.append('..')

from utils.accumulator import accumulator
from utils.dotdict import dotdict
from data.custom_dataset import Dataset_Custom
from data.data_loader import get_dataloader
pd.set_option('display.max_columns', 50)

torch.set_num_threads(4)

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


def predict(best_model, data_loader, config, y_scaler, Timerange):
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

    pred_df = pd.DataFrame(
        {'TimeStamp': Timerange,
         'test_y':torch.concat(inverse_test_metric.output, axis = 0).squeeze().cpu().numpy(), 
         'pred':torch.concat(inverse_test_metric.predict, axis = 0).squeeze().cpu().numpy()}
        , columns = ['TimeStamp','test_y','pred']
        )

    return pred_df

def PP_graph(test_data, stage_var_dic, config, y_scaler, save_model_path):

    seed = 2

    load_model = torch.load(save_model_path)
    model = load_model['model_structure']
    model.load_state_dict(load_model['best_model_state_dict_{}'.format(seed)])

    data_loader = {}
    data_loader['test'] = get_dataloader(test_data, stage_var_dic, config, shuffle = False, drop_last = False)

    pred_df = predict(model, data_loader, config, y_scaler, test_data.TimeStamp)

    pred_df = pred_df[(pred_df['TimeStamp'] >= '2021-08-22') & (pred_df['TimeStamp'] < '2021-08-25')].reset_index(drop=True)

    plt.figure(figsize = (8, 4), dpi = 600)

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.rc('xtick', labelsize = 8)
    plt.plot(pred_df.index.tolist(),pred_df['pred'].tolist(), marker = '',label='Predict',linestyle = '-',markersize = 1, color = 'blue', linewidth = 1)

    plt.plot(pred_df.index.tolist(),pred_df['test_y'].tolist(), marker = '',label='Label',linestyle = '--',markersize = 1, color = 'red', linewidth = 1)

    plt.ylabel('Pour Point [℃]')
    plt.xlabel('Index (Time)')

    plt.ylim(-14,-11)
    plt.legend(fontsize = 7,markerscale=0.5,loc='upper right')
    plt.tight_layout()
    #plt.savefig('pic/PP_pred_graph.png')
    plt.show()
    
def Kvis_graph(test_data, stage_var_dic, config, y_scaler, save_model_path):

    seed = 2

    load_model = torch.load(save_model_path)
    model = load_model['model_structure']
    model.load_state_dict(load_model['best_model_state_dict_{}'.format(seed)])

    data_loader = {}
    data_loader['test'] = get_dataloader(test_data, stage_var_dic, config, shuffle = False, drop_last = False)

    pred_df = predict(model, data_loader, config, y_scaler, test_data.TimeStamp)

    pred_df = pred_df[(pred_df['TimeStamp'] >= '2021-08-22') & (pred_df['TimeStamp'] < '2021-08-25')].reset_index(drop=True)

    plt.figure(figsize = (8, 4), dpi = 600)

    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    plt.rc('xtick', labelsize = 8)
    plt.plot(pred_df.index.tolist(),pred_df['pred'].tolist(), marker = '',label='Predict',linestyle = '-',markersize = 1, color = 'blue', linewidth = 1)

    plt.plot(pred_df.index.tolist(),pred_df['test_y'].tolist(), marker = '',label='Label',linestyle = '--',markersize = 1, color = 'red', linewidth = 1)

    plt.ylim(6.35,6.5)
    plt.ylabel('Kinematic Viscosity [cSt]')
    plt.xlabel('Index (Time)')


    plt.legend(fontsize = 7,markerscale=0.5,loc='upper right')
    plt.tight_layout()
    #plt.savefig('pic/Kvis_pred_graph.png')
    plt.show()


def make_dataset(data, configs, target, save_model_path):

    configs.target_character = target
    
    save_model_path = save_model_path + '{}_{}.tar.gz'.format(configs.model_name, configs.target_character)

    stage_var_dic['target'] = [configs.target_character]

    x_col, y_col = [i for j in stage_var_dic.values() for i in j if i not in stage_var_dic['target']], stage_var_dic['target']

    data = data[['TimeStamp'] + x_col + y_col]

    custom_dataset = Dataset_Custom(stage_var_dic, scale=True)

    train_data, val_data, test_data = custom_dataset.run(data)

    y_scaler = custom_dataset.y_scaler

    return test_data, y_scaler, save_model_path


# %%

if __name__ == "__main__":

    configs = dotdict()
    
    configs.model_name = 'MMPNet'
    configs.batch_size = 128
    configs.device_num = '5'

    data = pd.read_csv('../dataset/data.csv', parse_dates=['TimeStamp'])

    stage_var_dic = {'Stage1' : ['stage1_var1','stage1_var2','stage1_var3','stage1_var4','stage1_var5', 'stage1_var6','stage1_var7', 'stage1_var8', 'stage1_var9'],
                'Stage2' : ['stage2_var1'],
                'Stage3' : ['stage3_var1','stage3_var2','stage3_var3','stage3_var4', 'stage3_var5'],
                'Stage4' : ['stage4_var1', 'stage4_var2', 'stage4_var3','stage4_var4','stage4_var5', 'stage4_var6', 'stage4_var7', 'stage4_var8'],
                }
    
    save_model_path = '../exp/saved_model/'

    PP_data, PP_scaler, PP_model_path = make_dataset(data, configs, 'extension_PP', save_model_path)
    
    PP_graph(PP_data, stage_var_dic, configs, PP_scaler, PP_model_path)
    
    Kvis_data, Kvis_scaler, Kvis_model_path = make_dataset(data, configs, 'extension_Kvis', save_model_path)
    
    Kvis_graph(Kvis_data, stage_var_dic, configs, Kvis_scaler, Kvis_model_path)

# %%