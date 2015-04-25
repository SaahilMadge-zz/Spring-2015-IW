import pandas
import numpy as np

# read the data
df = pandas.read_csv('techsectordatareal.csv')
permnoList = sorted(set(list(df['PERMNO'])))
# for i in permnoList:
# 	print i
print len(permnoList) # should be 39
print permnoList

# for permno in permnoList:
# 	print "%d:\t%d" % (permno, list(df[df['PERMNO'] == permno]['date']).__contains__(20070103))

companiesNotFull = [12084, 13407, 14542, 93002] # companies without full dates

# read the sector data
ndxtdf = pandas.read_csv('ndxtdata.csv')
ndxtdf = ndxtdf.sort_index(by='Date', ascending=True)
# print ndxtdf
dates = ndxtdf['Date']
print dates
# print len(dates)
# print len(df[df['PERMNO'] == 10107])
print df[df['PERMNO'] == 10107]