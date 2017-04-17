import numpy as np
import pandas as pd
import MySQLdb
import sys
from db_info import db_info

from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN, GRU, Merge



class util:

    feat_table = 'features'

    def main(self):
        my_db_info = db_info()
        data_frame = self.retrieve_input(my_db_info)
        # for index , row in data_frame.iterrows():
        #     print index, ' , ', row


    def connectMysqlDB(self, db_info):
                conn = MySQLdb.connect(db_info.host, db_info.user, db_info.password, db_info.database)
                c = conn.cursor()
                return c


    def retrieve_input(self, db_info):
                print 'retrieve_input'
                data = []
                columns = []
                try:
                        self.cursor = self.connectMysqlDB(db_info)
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if(self.cursor is not None):
                                sql = "show columns from {0}".format(self.feat_table)
                                print 'sql: ' , sql
                                self.cursor.execute(sql)
                                columns_name = self.cursor.fetchall()
                                for row in columns_name:
                                        columns.append(row[0])
                                sql = "select * from {0}".format(self.feat_table)
                                self.cursor.execute(sql)
                                result = self.cursor.fetchall()
                                for row in result:
                                        data.append(row)

                print 'len(data):  '  , len(data) , '  ' ,  data[0]
                data_frame = pd.DataFrame(data = data, columns = columns)
                return data_frame


utility = util()
utility.main()