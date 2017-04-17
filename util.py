import numpy as np
import pandas as pd
import MySQLdb
import sys

from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU, Merge

class util:

    def connectMysqlDB(self):
                conn = MySQLdb.connect(self.host, self.user, self.password, self.database)
                c = conn.cursor()
                return c


    def retrieve_input(self):
                print 'retrieve_input'
                data = []
                columns = []
                try:
                        self.cursor = self.connectMysqlDB()
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if(self.cursor is not None):
                                sql = "show columns from {0}".format(self.trust_table)
                                self.cursor.execute(sql)
                                columns_name = self.cursor.fetchall()
                                for row in columns_name:
                                        columns.append(row[0])
                                sql = "select * from {0}".format(self.trust_table)
                                self.cursor.execute(sql)
                                result = self.cursor.fetchall()
                                for row in result:
                                        data.append(row)

                print 'len(data):  '  , len(data) , '  ' ,  data[0]
                data_frame = pd.DataFrame(data = data, columns = columns)
                #trust_data_frame = trust_data_frame.set_index('user_id')
                return data_frame
