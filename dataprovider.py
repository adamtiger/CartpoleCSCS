from enum import Enum
import pandas as pd


class Optimizer(Enum):

    ADAM = 1
    RMSPROP = 2
    SGD = 3


class Parameters:

    def __init__(self):

        self.id = None
        self.buf_size = None
        self.batch = None
        self.epoch = None
        self.max_ep = None
        self.eps = None
        self.gamma = None
        self.C = None
        self.train_freq = None
        self.eval_freq = None
        self.net = None
        self.opt = None
        self.log_file = None
        self.log_return_name = None
        self.log_loss_name = None
        self.w_file_name = None


class DataProvider:

    def __init__(self, file_name):

        self.df = pd.read_csv(file_name)
        self.num_records = len(self.df.index)

    def process_row(self, id):

        p = Parameters()

        p.id = id
        p.buf_size = self.df.iloc[id]['buf_size']
        p.batch = self.df.iloc[id]['batch']
        p.epoch = self.df.iloc[id]['epoch']
        p.max_ep = self.df.iloc[id]['max_ep']
        p.eps = self.df.iloc[id]['eps']
        p.gamma = self.df.iloc[id]['gamma']
        p.C = self.df.iloc[id]['C']
        p.train_freq = self.df.iloc[id]['train_freq']
        p.eval_freq = self.df.iloc[id]['eval_freq']

        n = self.df.iloc[id]['n']
        net = []
        for i in range(1, n + 1):
            units = self.df.iloc[id][('units' + str(i))]
            actvs = self.df.iloc[id][('activations' + str(i))]
            net.append((units, actvs))
        p.net = net

        opt = self.df.iloc[id]['opt']
        if str(opt).upper() == 'ADAM':
            p.opt = Optimizer.ADAM
        elif str(opt).upper() == 'RMSPROP':
            p.opt = Optimizer.RMSPROP
        elif str(opt).upper() == 'SGD':
            p.opt = Optimizer.SGD
        else:
            raise AttributeError('Unknown optimization method.')

        return p

