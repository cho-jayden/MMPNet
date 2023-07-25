# %%
import argparse
import pandas as pd

from utils.dotdict import dotdict
from exp.exp_MMP_Net import exp_main



# %%
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target')
    parser.add_argument('--model_name')
    parser.add_argument('--n_epoch')
    parser.add_argument('--learning_rate')
    parser.add_argument('--batch_size')
    parser.add_argument('--hidden_size')
    parser.add_argument('--activation')
    parser.add_argument('--dropout_p')
    parser.add_argument('--n_reg_layer')
    parser.add_argument('--device_num')

    args = parser.parse_args()

    return args

# %%

if __name__ == "__main__":

    args = get_argument_parser()

    configs = dotdict()

    configs.target = str(args.target)
    configs.num_workers = 4
    configs.model_name = str(args.model_name)
    configs.seed = None
    configs.device_num = str(args.model_name)
    configs.n_epoch = int(args.model_name)
    configs.learning_rate = float(args.model_name)
    configs.batch_size = int(args.model_name)
    configs.hidden_size = int(args.model_name)
    configs.activation = str(args.model_name)
    configs.dropout_p = float(args.model_name)
    configs.n_reg_layer = int(args.model_name)

    save_model_path = 'exp/saved_model/' + '{}_{}.tar.gz'.format(configs.model_name, configs.target)

    data = pd.read_csv('dataset/data.csv', parse_dates=['TimeStamp'])

    stage_var_dic = {'Stage1' : ['stage1_var1','stage1_var2','stage1_var3','stage1_var4','stage1_var5', 'stage1_var6','stage1_var7', 'stage1_var8', 'stage1_var9'],
                'Stage2' : ['stage2_var1'],
                'Stage3' : ['stage3_var1','stage3_var2','stage3_var3','stage3_var4', 'stage3_var5'],
                'Stage4' : ['stage4_var1', 'stage4_var2', 'stage4_var3','stage4_var4','stage4_var5', 'stage4_var6', 'stage4_var7', 'stage4_var8'],
                'target' : [configs.target]}

    x_col = [i for j in stage_var_dic.values() for i in j if i not in stage_var_dic['target']]
    y_col = stage_var_dic['target']

    data = data[['TimeStamp'] + x_col + y_col]

    exp_main(data, configs, stage_var_dic, save_model_path)

# %%