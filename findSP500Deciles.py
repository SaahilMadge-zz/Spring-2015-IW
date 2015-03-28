import pandas
import numpy
import matplotlib

df = pandas.read_csv('sp500marketcap.csv')

tsymList = sorted(list(set(df['TICKER'])))
print tsymList
print len(tsymList)
