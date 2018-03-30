'''
This module is responsible for
logging the most important information
into a file and create graphs if necessary.
'''

from enum import Enum
import csv


class Mode(Enum):

    STDOUT = 1
    LOG_F = 2   # log file
    LOSS_F = 3  # loss csv
    RET_F = 4   # return csv
    STD_LOG = 5 # both STDOUT and LOG file


class Logger:

    def __init__(self, params):
        self.log_file = open(params.log_file, 'a', buffering=1)
        self.loss_csv = open(params.log_loss_name, 'a', newline='', buffering=1)
        self.loss_csv_obj = csv.writer(self.loss_csv)
        self.return_csv = open(params.log_return_name, 'a', newline='', buffering=1)
        self.return_csv_obj = csv.writer(self.return_csv)

        self.log_funcs = [self.__log_STDOUT,
                          self.__log_LOG_F,
                          self.__log_LOSS_F,
                          self.__log_RET_F,
                          self.__log_STD_LOG
                          ]

    def log(self, mode, msg):
        success = False
        for func in self.log_funcs:
            success = success or func(mode, msg)

        if not success:
            raise AttributeError('Unknown mode ' + str(mode) + ' in logger!')

    def __del__(self):
        self.log_file.flush()
        self.loss_csv.flush()
        self.return_csv.flush()

        self.log_file.close()
        self.loss_csv.close()
        self.return_csv.close()

    # -----------------------------------
    # Private functions.

    def __log_STDOUT(self, mode, msg):

        if mode == Mode.STDOUT:
            print(msg)
            return True

        return False

    def __log_LOG_F(self, mode, msg):

        if mode == Mode.LOG_F:
            self.log_file.write(msg + '\n')
            return True

        return False

    def __log_LOSS_F(self, mode, msg):

        if mode == Mode.LOSS_F:  # msg: [iteration, episode, loss]
            self.loss_csv_obj.writerow(msg)
            return True

        return False

    def __log_RET_F(self, mode, msg):

        if mode == Mode.RET_F:  # msg: [iteration, episode, return]
            self.return_csv_obj.writerow(msg)
            return True

        return False

    def __log_STD_LOG(self, mode, msg):

        if mode == Mode.STD_LOG:
            print(msg)
            self.log_file.write(msg + '\n')
            return True

        return False
