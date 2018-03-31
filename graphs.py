import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os


def process_results_of_run(folder):
    
    def maximum(name):
        df = pd.read_csv(name, header=None, names=['itr', 'ep', 'rtn'])
        return df['rtn'].max(axis=0)

    path = folder + '/logfiles'
    files = os.listdir(path)
    num_fls = len(files)//3
    
    rtn_values = []
    for id in range(num_fls):
       name = path + '/' + str(id) + '_return_high.csv'
       rtn_values.append((id, maximum(name)))

    return rtn_values


def slicing(rtn_values):
    
    delta = 20
    max_rtn = 200

    groups = {}
    for i in range(max_rtn // delta):
        groups[i] = 0

    for rtn in rtn_values:
        id = rtn[1] // delta
        if id == 10:
            id = 9
        groups[id] += 1

    return groups


def draw_barplot(groups):

    ind = np.arange(len(groups.keys()))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plt.bar(ind, groups.values(), width)

    plt.xlabel('Returns')
    plt.ylabel('Number')
    plt.title('Distribution according to the returns')

    x_labels = ('0-20', '20-40', '40-60', '60-80', '80-100', '100-120',
               '120-140', '140-160', '160-180', '180-200')
    plt.xticks(ind, x_labels)
    plt.yticks(np.arange(0, max(groups.values()), max(groups.values())// 10 + 1))

    plt.show()


def create_bar_plot():

    x = process_results_of_run('run5_img')
    x = slicing(x)
    draw_barplot(x)


def create_learning_plot():
    folder = 'run5_img'
    x = process_results_of_run(folder)

    maximum = [0, 0]
    for e in x:
        if e[1] > maximum[1]:
            maximum[1] = e[1]
            maximum[0] = e[0]

    path = folder + '/logfiles/' + str(maximum[0]) + '_train_ret_high.csv'
    df = pd.read_csv(path, header=None, names=['itr', 'ep', 'rtn'])

    plt.xlabel('Episode')
    plt.ylabel('Returns')
    plt.title('Learning curve for high')

    plt.plot(df['ep'].as_matrix(), df['rtn'].as_matrix())
    plt.show()

#create_bar_plot()
create_learning_plot()