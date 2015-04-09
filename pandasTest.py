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

df = pandas.read_csv('sp500data.csv')

tsymList = sorted(list(set(df['TICKER'])))
print tsymList
companyChanges = {}
tsymList.pop(0)    #remove nan at the beginning
print len(tsymList)

f = open('pythonlist.txt', 'w+')

# es = list(df[df['TICKER'] == 'ES']['PRC'])
# print es

companiesToSkip = ['WBA'] # companies that shouldn't be in this data
for company in companiesToSkip:
	tsymList.remove(company)
# marketCapArray = numpy.array([])
marketCapArray = []
marketCapDict = {}

for i in range(len(tsymList)):
	if(tsymList[i] in companiesToSkip):
		continue

	compData = df[df['TICKER'] == tsymList[i]]
	marketShare = abs(list(compData['PRC'])[-5]) * abs(list(compData['SHROUT'])[-5])
	# print list(compData['PRC'])[-1]
	#f.write('%s:%f\n' % (tsymList[i], marketShare))
	print '%s: %f' % (tsymList[i], marketShare)
	marketCapArray.append(marketShare)
	marketCapDict[tsymList[i]] = [marketShare]

# 	companyChanges[tsymList[i]] = []

# print marketCapArray
for i in range(1,11):
	print "%d percentile: %f" % (i*10, numpy.percentile(marketCapArray, i*10))

for company in tsymList:
	marketCap = marketCapDict[company][0]
	percentile = findPercentile(marketCap, marketCapArray)
	print '%s: %f\tpercentile:%d' % (company, marketCap, percentile)
	marketCapDict[company].append(percentile)

# print numpy.percentile(marketCapArray, 10)

# # msft = df[df['TSYMBOL'] == 'MSFT']
# # msftPrice = msft['PRC']
alpha = 0.15  # percent change needed to be significant

standardDeviationArray = []
for i in range(10):
	standardDeviationArray.append([])

meanReturnArray = []
for i in range(10):
	meanReturnArray.append([])

def findChangeIndices(companyName):
	
	priceList = numpy.array(list(df[df['TICKER'] == str(companyName)]['PRC']))
	priceList = priceList[~numpy.isnan(priceList)]
	# print priceList
	# print priceList[0]
	# print priceList
	changeList = []

	# # msftChangeList = []
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

	# #print changeList
	# companyChanges[companyName] = changeList

# # print msftChangeList

# # findChangeIndices("EBAY")
# # print companyChanges

# print marketCapDict['AA']

def findMean(companyName):
	priceList = numpy.array(list(df[df['TICKER'] == str(companyName)]['PRC']))
	priceList = priceList[numpy.logical_not(numpy.isnan(priceList))]
	return numpy.mean(priceList)

def findStdDev(companyName):
	priceList = numpy.array(list(df[df['TICKER'] == str(companyName)]['PRC']))
	priceList = priceList[numpy.logical_not(numpy.isnan(priceList))]
	return numpy.std(priceList)

percentileChangesArray = [0]*10

for companyName in tsymList:
	# get market cap, find number of changes, and also find standard deviation and mean return
    percentile = marketCapDict[companyName][1]
    numChanges = findChangeIndices(companyName)
    mean = findMean(companyName)
    stdDev = findStdDev((companyName))
    print "%s: %d\t%d changes\t %f mean\t %f std" % (companyName, percentile, numChanges, mean, stdDev)
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

# plot
decileArray = []
for i in range(10):
	decileArray.append('%d decile' % (i+1))

x_pos = numpy.arange(len(decileArray))
# graph of percentile changes by decile
plt.figure(1)
plt.bar(x_pos, percentileChangesArray)
plt.xticks(x_pos, decileArray)
plt.ylabel('Days with more than %f%% change in stock price' % (alpha*100))

# graph of standard deviation by decile
plt.figure(2)
plt.subplot(2,1,1)
plt.bar(x_pos, standardDeviationArray)
plt.xticks(x_pos, decileArray)
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs stock decile')
plt.subplot(2,1,2)
plt.plot(standardDeviationArray, meanReturnArray, 'ro')
#plt.xticks(standardDeviationArray, decileArray)
plt.ylabel('Mean Return')
plt.title('Return vs Standard Deviation')

plt.show()

# for companyName in tsymList:
#     if(len(companyChanges[companyName]) > 10):
#         print "company: %s\t times:%d" % (companyName,len(companyChanges[companyName]))

# # findChangeIndices('AIG')

# # print companyChanges
# sumTerm = 0
# for key in companyChanges:
# 	sumTerm = sumTerm + len(companyChanges[key])

# print "sum: %d" % sumTerm
