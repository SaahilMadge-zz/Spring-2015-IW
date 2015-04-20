import pandas
import numpy as np
import matplotlib
from scipy import stats

def getCurrencyData():
	df = pandas.read_csv('basicCurrencyData.csv')

	countries = np.delete(df.columns.values, 0)
	print countries
	# test = df[countries[0]]
	# test = list(test[np.logical_not(np.isnan(test))])
	# print test

	countryRateChangeDict = {}
	for country in countries:
		fxRates = df[country]
		fxRates = list(fxRates[np.logical_not(np.isnan(fxRates))])

		alpha = 0.05  # percent change needed to be significant

		changeList = []

		lastRate = abs(fxRates[0])
		counter = 0
		for rate in fxRates:
			absRate = abs(rate)
			# print "price:%f\t%r" % (absPrice, absPrice < (1-alpha)*lastPrice or absPrice > (1+alpha)*lastPrice)
			if (absRate < (1-alpha)*lastRate or absRate > (1+alpha)*lastRate):
				counter += 1
				# if (country == 'china'):
				# 	print "last:%s  new:%s" % (absRate, lastRate)
			lastRate = absRate

		countryRateChangeDict[country] = counter

	print countryRateChangeDict
	print sum(countryRateChangeDict.values())