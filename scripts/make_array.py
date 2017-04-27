import sys
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
import numpy as np
import pandas as pd
import pickle

db = "mztwitter"
myDB = URL(drivername='mysql', database = db, query={ 'read_default_file' : '/home/adarsh/.my.cnf'})
engine = create_engine(name_or_url=myDB)
connection = engine.connect()


saf_cnty = 60
ip_cnty = 72
ip_msp = 78

def add_nulls(connection):	
	months = 45;
	
	sel = connection.execute("select distinct(cnty) from NLP_features_msp")
	DF = pd.DataFrame(sel.fetchall())	
	
	counties = DF.values

	print counties.shape

	for county in counties:
		for i in range(months):
			qr = 'insert ignore into NLP_features_msp values( '
		        qr += '\''+county +'_'+ str(i) +'\' , '
		        for j in xrange(ip_msp):
		                qr += ' NULL , '
        		qr += '\''+county + '\' , ' + str(i) + ' )'
			#print qr[0]
			connection.execute(qr[0])

def make_data(connection):
	sel = connection.execute("select * from NLP_features_saf")
	df = pd.DataFrame(sel.fetchall())
	df.columns = sel.keys()
	
	# removing these three columns and keeping just the values
	df.drop('cnty_month', axis=1, inplace=True)
	df.drop('month', axis=1, inplace=True)
	df.drop('cnty', axis=1, inplace=True)
	
	array = df.values	
	cnty_month_total = array.shape[0]
		
	array = np.reshape(array, (cnty_month_total / 45, 45, saf_cnty))
	print "New shape : ",array.shape
	return array;
	
#add_nulls(connection)
make_data(connection)
