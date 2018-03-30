import argparse
import multiprocessing as mp
import dataprovider as dp
import dqn_train as dqn

parser = argparse.ArgumentParser(description='DQN for CartPole')

parser.add_argument('--mode', default=0, type=int, metavar='N',
        help='the mode of the run. 0: dqn_low, 1: dqn_high')
parser.add_argument('--trds', default=2, type=int, metavar='N',
        help='the number of processes')

args = parser.parse_args()


# ------------------------------------------
# Folder names:
log_folder = "logfiles/"
weight_folder = "weights/"

# Files with paramters:
low_params = "low_params.csv"
high_params = "high_params.csv"


def mp_start(msg, func, data):

    print(msg)

    # Get the data
    inputs = data()

    print('Start multiprocessing.')
    pool = mp.Pool(processes=args.trds)
    pool.map(func, inputs)
    pool.close()
    pool.join()

    print('----- Finished! -----')


# ------------------------------------------
# Mode 1: CartPole with low level state

def data_for_low():

    provider = dp.DataProvider(low_params)

    data = []

    for id in range(provider.num_records):
        p = provider.process_row(id)
        p.log_file = log_folder + str(id) + "_log_low.log"
        p.log_loss_name = log_folder + str(id) + "_loss_low.csv"
        p.log_return_name = log_folder + str(id) + "_return_low.csv"
        p.w_file_name = weight_folder + str(id) + "weights_low.h5"

        data.append(p)
    return data


def process_of_low(params):
    q = dqn.DqnLow(params)
    q.train_function()


# ------------------------------------------
# Mode 2: CartPole with high level state

def data_for_high():
    provider = dp.DataProvider(high_params)

    data = []

    for id in range(provider.num_records):
        p = provider.process_row(id)
        p.log_file = log_folder + str(id) + "_log_high.log"
        p.log_loss_name = log_folder + str(id) + "_loss_high.csv"
        p.log_return_name = log_folder + str(id) + "_return_high.csv"
        p.w_file_name = weight_folder + str(id) + "weights_high.h5"

        data.append(p)
    return data


def process_of_high(params):
    q = dqn.DqnLow(params)
    q.train_function()

if __name__ == "__main__":

    if args.mode == 0:
        mp_start("---- Low: Multiprocessing ----", process_of_low, data_for_low)
    elif args.mode == 1:
        mp_start("---- High: Multiprocessing ----", process_of_high, data_for_high)
    else:
        raise ValueError('Mode with number: (' + str(args.mode) + ') does not exist!')


