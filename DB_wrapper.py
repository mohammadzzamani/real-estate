import numpy as np
import pandas as pd
import MySQLdb
import sys
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

import DB_info





class DB_wrapper:

    def __init__(self):
        self.connectMysqlDB()



    def connectMysqlDB(self):
                myDB = URL(drivername='mysql', database=DB_info.DB, query={'read_default_file' : DB_info.CONF_FILE })
                self.engine = create_engine(name_or_url=myDB)
                # connection = engine.connect()
                # conn = MySQLdb.connect(db_info.host, db_info.user, db_info.password, db_info.database)
                # c = conn.cursor()
                # return engine


    def retrieve_data(self, table):
                print 'retrieve_data'
                try:
                        connection = self.engine.connect()
                        # self.cursor = self.connectMysqlDB(db_info)
                except:
                        print("error while connecting to database:", sys.exc_info()[0])
                        raise
                if connection is not None:
                                columns = []
                                sql = "show columns from {0}".format(table)
                                print 'sql: ' , sql
                                query = connection.execute(sql)
                                columns_name = query.fetchall()
                                for row in columns_name:
                                        columns.append(row[0])
                                sql = "select * from {0}".format(table)
                                query = connection.execute(sql)
                                result = query.fetchall()
                                data_frame = pd.DataFrame(data = result, columns = columns)
                return data_frame
