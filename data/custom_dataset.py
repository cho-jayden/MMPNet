import numpy as np
from sklearn.preprocessing import StandardScaler

class Dataset_Custom():
    def __init__(self, stage_vars, data_split_ratio = (0.8, 0.1, 0.1), scale = True):

        self.stage_vars = stage_vars

        self.split_train = data_split_ratio[0]
        self.split_val = data_split_ratio[1]
        self.split_test = data_split_ratio[2]

        self.scale = scale
        self.x_scaler = None
        self.y_scaler = None


    def NullOutlier(self, df, target_list):
        for i in target_list:
            df.loc[df[df[i].isnull() == True].index, 'outlier'] = 1

        df = df[df['outlier'] != 1].reset_index(drop = True)
        df = df.drop(columns = 'outlier')
        return df

    def StdOutlierDetecting(self, df, Target_list, stdNUM) :
        outlierIndex_list = []
        for t in Target_list: 
            mean = np.mean(df[t])
            std = np.std(df[t])
            lower_bound = mean - stdNUM * std
            upper_bound = mean + stdNUM * std

            low_df = df[df[t] < lower_bound]
            up_df = df[df[t] > upper_bound]

            currentLow_list = list(low_df.index)
            currentUp_list = list(up_df.index)
            outlierIndex_list += sorted(currentLow_list + currentUp_list)
            outlierIndex_list = list(set(outlierIndex_list))

        df.loc[outlierIndex_list, 'outlier'] = 1

        df = df[df['outlier'] != 1].reset_index(drop = True)
        df = df.drop(columns = 'outlier')

        return df

    def fit_scaler(self, train_data):

        train_data_temp = train_data.copy()

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x_col = [i for j in self.stage_vars.values() for i in j if i not in self.stage_vars['target']]
        y_col = self.stage_vars['target']

        train_data_x = train_data_temp[x_col]
        train_data_y = train_data_temp[y_col]
        
        x_scaler.fit(train_data_x)
        y_scaler.fit(train_data_y)

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler


    def transform_scaler(self, data):

        temp_data = data.copy()

        x_col = [i for j in self.stage_vars.values() for i in j if i not in self.stage_vars['target']]
        y_col = self.stage_vars['target']

        temp_data.loc[:, x_col] = self.x_scaler.transform(temp_data.loc[:, x_col])
        temp_data.loc[:, y_col] = self.y_scaler.transform(temp_data.loc[:, y_col])

        return temp_data

    def run(self, data):

        Null_check_col = [i for j in self.stage_vars.values() for i in j]
        data = self.NullOutlier(data, Null_check_col)

        Std_check_col = [i for j in self.stage_vars.values() for i in j if i not in ['time_var']]
        data = self.StdOutlierDetecting(data, Std_check_col, 4)

        num_train = int(len(data) * self.split_train)
        num_test = int(len(data) * self.split_test)
        num_vali = len(data) - num_train - num_test

        train_data = data[:num_train]
        val_data = data[num_train :num_train + num_vali]
        test_data = data[num_train + num_vali:]

        if self.scale:
            self.fit_scaler(train_data)

            train_data = self.transform_scaler(train_data)
            val_data = self.transform_scaler(val_data)
            test_data = self.transform_scaler(test_data)

            return train_data, val_data, test_data

        else:
            return train_data, val_data, test_data