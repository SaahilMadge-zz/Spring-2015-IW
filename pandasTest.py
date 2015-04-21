import pandas
import numpy
import matplotlib.pyplot as plt
from scipy import stats

def findPercentile(marketCap, marketCapArray):
	for i in range(1,11):
		percentile = numpy.percentile(marketCapArray, i*10)
		if (marketCap < percentile):
			return i
	return 10  # if it's the highest

df = pandas.read_csv('sp500data2.csv')

permnoList = sorted(list(set(df['PERMNO'])))
# skip certain companies b/c their price data is not full
#permnoToSkip = [10942, 11896]
#permnoList = [x for x in permnoList if x not in permnoToSkip]
print permnoList
companyChanges = {}
#permnoList.pop(0)    #remove nan at the beginning
print len(permnoList)

f = open('pythonlist.txt', 'w+')

#companiesToSkip = ['WBA'] # companies that shouldn't be in this data
#for company in companiesToSkip:
#	permnoList.remove(company)
# marketCapArray = numpy.array([])
marketCapArray = []
marketCapDict = {}

for i in range(len(permnoList)):
	# if(permnoList[i] in companiesToSkip):
		# continue

	compData = df[df['PERMNO'] == permnoList[i]]
	marketShare = abs(list(compData['PRC'])[-5]) * abs(list(compData['SHROUT'])[-5])
	# print list(compData['PRC'])[-1]
	#f.write('%s:%f\n' % (permnoList[i], marketShare))
	print '%s: %f' % (permnoList[i], marketShare)
	marketCapArray.append(marketShare)
	marketCapDict[permnoList[i]] = [marketShare]

# print marketCapArray
for i in range(1,11):
	print "%d percentile: %f" % (i*10, numpy.percentile(marketCapArray, i*10))

for permno in permnoList:
	marketCap = marketCapDict[permno][0]
	percentile = findPercentile(marketCap, marketCapArray)
	print '%s: %f\tpercentile:%d' % (permno, marketCap, percentile)
	marketCapDict[permno].append(percentile)

alpha = 0.15  # percent change needed to be significant

standardDeviationArray = []
for i in range(10):
	standardDeviationArray.append([])

meanReturnArray = []
for i in range(10):
	meanReturnArray.append([])

def findChangeIndices(permno):
	
	priceList = numpy.array(list(df[df['PERMNO'] == permno]['PRC']))
	priceList = priceList[~numpy.isnan(priceList)]

	changeList = []

	lastPrice = abs(priceList[0])
	for i in range(len(priceList)):
		absPrice = abs(priceList[i])
		if (absPrice == 0):
			absPrice = lastPrice
		# print "price:%f\t%r" % (absPrice, absPrice < (1-alpha)*lastPrice or absPrice > (1+alpha)*lastPrice)
		if (absPrice < (1-alpha)*lastPrice or absPrice > (1+alpha)*lastPrice):
			changeList.append(i)
		lastPrice = absPrice

	return len(changeList)

def isFloat(x):
	try:
		a = float(x)
	except ValueError:
		return False
	else:
		return True

def findMeanAndStdDev(permno):
	# get returns and find return
	#priceList = numpy.array(list(df[df['PERMNO'] == permno]['PRC']))
	# print type(returnList)
	returnList = numpy.array(list(df[df['PERMNO'] == permno]['RET']))
	returnList = numpy.array([x for x in returnList if isFloat(x)])
	#print returnList
	# print type(returnList2)
	returnList = returnList.astype('float')
	returnList = returnList[numpy.logical_not(numpy.isnan(returnList))]

	# find the % return

	# get the mean as a %
	percentageMean = numpy.mean(returnList)#/float(priceList[0])
	percentageStdDev = numpy.std(returnList)
	return (percentageMean, percentageStdDev)

# def findStdDev(permno):
# 	priceList = numpy.array(list(df[df['PERMNO'] == permno]['PRC']))
# 	priceList = priceList[numpy.logical_not(numpy.isnan(priceList))]

# 	# get the stddev as a %
# 	percentageStdDev = numpy.std(priceList)#/float(priceList[0])
# 	return percentageStdDev

percentileChangesArray = [0]*10

for permno in permnoList:
	# get market cap, find number of changes, and also find standard deviation and mean return
    percentile = marketCapDict[permno][1]
    numChanges = findChangeIndices(permno)
    meanAndStdDev = findMeanAndStdDev(permno)
    mean = meanAndStdDev[0]
    stdDev = meanAndStdDev[1]
    #mean = findMean(permno)
    # stdDev = findStdDev((permno))
    print "%s: %d\t%d changes\t %f mean\t %f std" % (permno, percentile, numChanges, mean, stdDev)
    percentileChangesArray[percentile-1] += numChanges
    standardDeviationArray[percentile-1].append(stdDev)
    meanReturnArray[percentile-1].append(mean)

print percentileChangesArray
print sum(percentileChangesArray)

for i in range(10):
	standardDeviationArray[i] = numpy.mean(standardDeviationArray[i])
	meanReturnArray[i] = numpy.mean(meanReturnArray[i])

print "standardDeviationArray"
print standardDeviationArray
print "meanReturnArray"
print meanReturnArray


# countryRateChangeDict = {}
# def getCurrencyData():
# 	global countryRateChangeDict
# 	df2 = pandas.read_csv('basicCurrencyData.csv')

# 	countries = numpy.delete(df2.columns.values, 0)
# 	print countries
# 	# test = df[countries[0]]
# 	# test = list(test[np.logical_not(np.isnan(test))])
# 	# print test

# 	for country in countries:
# 		fxRates = df2[country]
# 		fxRates = list(fxRates[numpy.logical_not(numpy.isnan(fxRates))])

# 		alpha = 0.05  # percent change needed to be significant

# 		changeList = []

# 		lastRate = abs(fxRates[0])
# 		counter = 0
# 		for rate in fxRates:
# 			absRate = abs(rate)
# 			# print "price:%f\t%r" % (absPrice, absPrice < (1-alpha)*lastPrice or absPrice > (1+alpha)*lastPrice)
# 			if (absRate < (1-alpha)*lastRate or absRate > (1+alpha)*lastRate):
# 				counter += 1
# 				# if (country == 'china'):
# 				# 	print "last:%s  new:%s" % (absRate, lastRate)
# 			lastRate = absRate

# 		countryRateChangeDict[country] = counter

# 	print countryRateChangeDict
# 	print sum(countryRateChangeDict.values())

# getCurrencyData()

# plot
decileArray = []
for i in range(10):
	decileArray.append('%d' % (i+1))

# graph of percentile changes by decile
plt.figure(1)
# # add the currency data to it
# percentileChangesArray.append(sum(countryRateChangeDict.values()))
# decileArray.append("FX rates")
x_pos = numpy.arange(len(decileArray))

print len(x_pos)
print len(percentileChangesArray)

plt.bar(x_pos, percentileChangesArray)
plt.xticks(x_pos, decileArray)
plt.ylabel('Days with more than %f%% change in price' % (alpha*100))

# graph of standard deviation by decile
plt.figure(2)
plt.subplot(2,1,1)
# # add currency data to it
# standardDeviationArray.append(numpy.std(countryRateChangeDict.values()))
# meanReturnArray.append(numpy.mean(countryRateChangeDict.values()))
plt.bar(x_pos, standardDeviationArray)
plt.xticks(x_pos, decileArray)
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs stock decile')
plt.subplot(2,1,2)
plt.plot(standardDeviationArray, meanReturnArray, 'ro')
plt.xticks(standardDeviationArray, decileArray)
plt.ylabel('Mean Return')
plt.title('Return vs Standard Deviation')

plt.show()
