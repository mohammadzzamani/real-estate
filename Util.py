import numpy as np
import pandas as pd
import MySQLdb
import sys
from DB_info import DB_info





class Util:

    tables = ['features', 'features']

    def main(self):
        my_db_info = DB_info()
        train_data , test_data = self.retrieve_input(my_db_info)
        # for index , row in data_frame.iterrows():
        #     print index, ' , ', row
        return  train_data , test_data

    def connectMysqlDB(self, db_info):
                conn = MySQLdb.connect(db_info.host, db_info.user, db_info.password, db_info.database)
                c = conn.cursor()
                return c


    def retrieve_input(self, db_info):
                print 'retrieve_input'

                all_data = []
                try:
                        self.cursor = self.connectMysqlDB(db_info)
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if(self.cursor is not None):
                    for table in self.tables:
                                data = []
                                columns = []
                                sql = "show columns from {0}".format(table)
                                print 'sql: ' , sql
                                self.cursor.execute(sql)
                                columns_name = self.cursor.fetchall()
                                for row in columns_name:
                                        columns.append(row[0])
                                sql = "select * from {0}".format(table)
                                self.cursor.execute(sql)
                                result = self.cursor.fetchall()
                                for row in result:
                                        data.append(row)

                                print 'len(data):  '  , len(data) , '  ' ,  data[0]
                                data_frame = pd.DataFrame(data = data, columns = columns)
                                all_data.append(data_frame)
                return all_data


utility = Util()
utility.main()