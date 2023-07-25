import torch
import torch.nn as nn

class MMP_layer(torch.nn.Module):
    def __init__(self, config, stage_dim):
        super(MMP_layer, self).__init__()

        self.MMPlayer = nn.Sequential(
            nn.Linear(stage_dim, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            getattr(nn, config.activation)(),
            nn.Dropout(p = config.dropout_p)
        )

    def forward(self, x):

        x = self.MMPlayer(x)

        return x

class Regressor_layer(torch.nn.Module):
    def __init__(self, config):
        super(Regressor_layer, self).__init__()

        hidden_size = config.hidden_size
        activation = config.activation
        n_reg_layer = config.n_reg_layer

        regressor_modules = []
        
        in_hidden = hidden_size
        out_hidden = hidden_size // 2
        for i in range(n_reg_layer):
            regressor_modules.append(nn.Linear(in_hidden, out_hidden))
            regressor_modules.append(getattr(nn, activation)())

            in_hidden = out_hidden
            out_hidden = out_hidden // 2

        regressor_modules.append(nn.Linear(in_hidden, 1, bias = True))

        self.regressor = nn.Sequential(*regressor_modules)

    def forward(self, x):

        x = self.regressor(x)
        
        return x

class MMPNET(torch.nn.Module):
    def __init__(self, config, stage_var_dic):
        super(MMPNET, self).__init__()

        self.stage_n = len([i for i in stage_var_dic.keys() if i != 'target'])

        self.stage_inputs_length = []
        for stage_key in [i for i in stage_var_dic.keys() if i != 'target']:
            self.stage_inputs_length.append(len(stage_var_dic[stage_key]))

        MMP_layer_list = []
        for stage_num in range(self.stage_n):
            if stage_num == 0:
                stage_dim = self.stage_inputs_length[stage_num]
            else:
                stage_dim = self.stage_inputs_length[stage_num] + config.hidden_size
            
            MMP_layer_list.append(MMP_layer(config, stage_dim))

        self.MMP_layer_list = nn.ModuleList(MMP_layer_list)

        self.regressor = Regressor_layer(config)
        
            
    def forward(self, x):

        stage_input_list = []
        start, end = 0, self.stage_inputs_length[0]
        for i in range(1, self.stage_n):
            stage_input_list.append(x[:,start:end])
            start,end = end, end + self.stage_inputs_length[i]
        stage_input_list.append(x[:,start:end])

        for i in range(self.stage_n):
            if i == 0:
                x = stage_input_list[i]
            else:
                x = torch.cat([x, stage_input_list[i]], dim = 1)
            x = self.MMP_layer_list[i](x)

        x = self.regressor(x)
        
        return x

