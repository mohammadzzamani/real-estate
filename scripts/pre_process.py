import sys
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import numpy as np
import pandas as pd
import load_DB as ld
import pickle

db = "mztwitter"
myDB = URL(drivername='mysql', database = db, query={ 'read_default_file' : '/home/adarsh/.my.cnf'})
engine = create_engine(name_or_url=myDB)
connection = engine.connect()

# List of all the years and month that will be used for training
train = ['2011_11', '2011_12', '2012_01', '2012_02', '2012_03', '2012_04', '2012_05', '2012_06', '2012_07', '2012_08','2012_09', '2012_10', '2012_11', '2012_12','2013_01', '2013_02', '2013_03', '2013_04', '2013_05', '2013_06', '2013_07', '2013_08','2013_09', '2013_10', '2013_11', '2013_12', '2014_01', '2014_02', '2014_03', '2014_04', '2014_05', '2014_06', '2014_07']

# List of all the years and months that will be used for testing
test = ['2014_08','2014_09', '2014_10', '2014_11', '2014_12','2015_01', '2015_02', '2015_03', '2015_04', '2015_05', '2015_06', '2015_07']


# Drop the tables if they exist
#connection.execute("drop table if exists NLP_train_data")
connection.execute("drop table if exists NLP_test_data")

# Create a table for train
#connection.execute("create table if not exists NLP_train_data (message_id bigint(20), cnty int(5), message text, created_time timestamp) CHARSET=utf8, COLLATE=utf8_unicode_ci")

'''
print "Starting random selection of train data"
for month in train:
        table = 'msgs_' + month
        print
        print "Table currently executing : ", table
        connection.execute('insert into NLP_train_data select message_id, cnty, message, created_time from ( select  message_id, cnty, message, created_time, CASE WHEN @cnty != cnty THEN @rn := 1 ELSE @rn := @rn + 1 END rn,@cnty:=cnty FROM (select * FROM %s ORDER BY RAND()) a, (select @rn := 0, @cnty:= NULL) r ORDER BY cnty) s WHERE (case when (rn * 0.1) <= 200 then rn <= LEAST(200, rn) else rn <= LEAST(1000, rn * 0.1) end)' % table)
'''
print
print "Starting random selection of test data"
# Create a table for test
connection.execute("create table if not exists NLP_test_data (message_id bigint(20), cnty int(5), message text, created_time timestamp) CHARSET=utf8, COLLATE=utf8_unicode_ci")

for month in test:
        table = 'msgs_' + month
        print
        print "Table currently executing : ", table
        connection.execute('insert into NLP_test_data select message_id, cnty, message, created_time from ( select  message_id, cnty, message, created_time, CASE WHEN @cnty != cnty THEN @rn := 1 ELSE @rn := @rn + 1 END rn,@cnty:=cnty FROM (select * FROM %s ORDER BY RAND()) a, (select @rn := 0, @cnty:= NULL) r ORDER BY cnty) s WHERE (case when (rn * 0.1) <= 200 then rn <= LEAST(200, rn) else rn <= LEAST(1000, rn * 0.1) end)' % table)

print "Completed! Bye"


