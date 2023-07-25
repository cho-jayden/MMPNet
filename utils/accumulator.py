import torch
import torch.nn as nn

class accumulator():
    def __init__(self):
        self.cnt = 0
        self.obs = 0
        self.loss = 0

        self.output = []
        self.predict = []

    def add(self, predict, output):

        assert len(predict) == len(output)

        self.output.append(output)
        self.predict.append(predict)
        self.obs += len(predict)
    
    def loss_update(self, loss, batch_size):
        self.loss += loss * batch_size

    def loss_stat(self):
        if self.obs != 0:
            run_loss = self.loss / self.obs
        else:
            run_loss = 0

        return run_loss

    def running_stat(self):
        if self.obs != 0:
            run_loss = self.loss / self.obs
        else: 
            run_loss = 0

        return run_loss

    def running_metric(self):
        output_list = torch.cat(self.output)
        predict_list = torch.cat(self.predict)

        r2 = self.r2_loss(predict_list, output_list).item()
        mse = torch.nn.MSELoss()(predict_list, output_list).item()
        rmse = mse ** 0.5
        mae = torch.nn.L1Loss()(predict_list, output_list).item()

        return round(mse, 4), round(rmse, 4), round(mae, 4), round(r2, 5)

    def reset(self):
        self.__init__()


    def r2_loss(self, predict, output):
        target_mean = torch.mean(output)
        ss_tot = torch.sum((output - target_mean) ** 2)
        ss_res = torch.sum((output - predict) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2