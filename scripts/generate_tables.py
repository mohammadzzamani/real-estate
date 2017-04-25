from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import pandas as pd
import numpy as np
import sys

X = np.loadtxt("/home/adarsh/NLP/test_features_ip")

db = "mztwitter"
county = []

with open("/home/adarsh/NLP/test_features_cnty_ip", "rb") as file:
        for line in file:
                county.append(line.rstrip('\n'))

print("Size of cnty col : ", len(county))
print('Shape of X : ', X.shape)

myDB = URL(drivername='mysql', database=db, query={
           'read_default_file' : '/home/adarsh/.my.cnf' })
engine = create_engine(name_or_url=myDB)
connection = engine.connect()

feature_names = [str("feat_"+ str(i)) for i in xrange(X.shape[1])]

feature_str = ''
for name in feature_names:
        feature_str += ', '+ name + ' float default null '

connection.execute('drop table if exists NLP_test_features_ip')

qry = 'create table NLP_test_features_ip (cnty_month varchar(15) primary key {0} )'.format(feature_str)
query = connection.execute(qry)

# Making the query to insert data into DB
for i in range(0, X.shape[0]):
        qry = 'insert into NLP_test_features_ip values ( '
        qry += '\''+ str(county[i]) + '\''
        for j in xrange(len(X[i])):
                qry += ', '+ str(X[i][j])
        qry += ')'
        if (i == 0):
                print(qry)
        connection.execute(qry)
print("Writing to DB complete! Bye")

