import pandas
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation

# read the data
df = pandas.read_csv('techsectordatareal.csv')
daysAhead = 270

# calculate price volatility array given company
def calcPriceVolatility(numDays, priceArray):
	global daysAhead
	# make price volatility array
	volatilityArray = []
	movingVolatilityArray = []
	for i in range(1, numDays+1):
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
	volatilityArray.append(np.mean(movingVolatilityArray))
	for i in range(numDays + 1, len(priceArray) - daysAhead):
		del movingVolatilityArray[0]
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
		volatilityArray.append(np.mean(movingVolatilityArray))

	return volatilityArray

# calculate momentum array
def calcMomentum(numDays, priceArray):
	global daysAhead
	# now calculate momentum
	momentumArray = []
	movingMomentumArray = []
	for i in range(1, numDays + 1):
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
	momentumArray.append(np.mean(movingMomentumArray))
	for i in range(numDays+1, len(priceArray) - daysAhead):
		del movingMomentumArray[0]
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
		momentumArray.append(np.mean(movingMomentumArray))

	return momentumArray

def makeModelAndPredict(permno, numDays, sectorVolatility, sectorMomentum, splitNumber):
	global df
	global daysAhead
	# get price volatility and momentum for this company
	companyData = df[df['PERMNO'] == permno]
	companyPrices = list(companyData['PRC'])

	volatilityArray = calcPriceVolatility(numDays, companyPrices)
	momentumArray = calcMomentum(numDays, companyPrices)

	splitIndex = splitNumber - numDays

	# since they are different lengths, find the min length
	if len(volatilityArray) > len(sectorVolatility):
		difference = len(volatilityArray) - len(sectorVolatility)
		del volatilityArray[:difference]
		del momentumArray[:difference]

	elif len(sectorVolatility) > len(volatilityArray):
		difference = len(sectorVolatility) - len(volatilityArray)
		del sectorVolatility[:difference]
		del sectorMomentum[:difference]

	# create the feature vectors X
	X = np.transpose(np.array([volatilityArray, momentumArray, sectorVolatility, sectorMomentum]))

	# create the feature vectors Y
	Y = []
	for i in range(numDays, len(companyPrices) - daysAhead):
		Y.append(1 if companyPrices[i+daysAhead] > companyPrices[i] else -1)
	print len(Y)

	# fix the length of Y if necessary
	if len(Y) > len(X):
		print 'here2'
		difference = len(Y) - len(X)
		del Y[:difference]

	# split into training and testing sets
	X_train = np.array(X[0:splitIndex]).astype('float64')
	X_test = np.array(X[splitIndex:]).astype('float64')
	y_train = np.array(Y[0:splitIndex]).astype('float64')
	y_test = np.array(Y[splitIndex:]).astype('float64')

	# fit the model and calculate its accuracy
	rbf_svm = svm.SVC(kernel='rbf')
	rbf_svm.fit(X_train, y_train)
	score = rbf_svm.score(X_test, y_test)
	print score
	return score

def main():
	global df

	# find the list of companies
	permnoList = sorted(set(list(df['PERMNO'])))
	companiesNotFull = [12084, 13407, 14542, 93002, 15579] # companies without full dates

	# read the tech sector data
	ndxtdf = pandas.read_csv('ndxtdata.csv')
	ndxtdf = ndxtdf.sort_index(by='Date', ascending=True)
	ndxtPrices = list(ndxtdf['Close'])

	# find when 2012 starts
	startOfTwelve = list(df[df['PERMNO'] == 10107]['date']).index(20120103)

	# we want to predict where it will be on the next day based on X days previous
	numDaysArray = [5, 10, 20, 90, 270] # day, week, month, quarter, year

	predictionDict = {}

	# iterate over combinations of n_1 and n_2 and find prediction accuracies
	for numDayIndex in numDaysArray:
		for numDayStock in numDaysArray:
			ndxtVolatilityArray = calcPriceVolatility(numDayIndex, ndxtPrices)
			ndxtMomentumArray = calcMomentum(numDayIndex, ndxtPrices)
			predictionForGivenNumDaysDict = {}

			for permno in permnoList:
				if permno in companiesNotFull:
					continue
				print permno
				percentage = makeModelAndPredict(permno,numDayStock,ndxtVolatilityArray,ndxtMomentumArray,startOfTwelve)
				predictionForGivenNumDaysDict[permno] = percentage


			predictionAccuracies = predictionForGivenNumDaysDict.values()
			meanAccuracy = np.mean(predictionAccuracies)
			maxIndex = max(predictionForGivenNumDaysDict, key=predictionForGivenNumDaysDict.get)
			maxAccuracy = (maxIndex, predictionForGivenNumDaysDict[maxIndex])
			minIndex = min(predictionForGivenNumDaysDict, key=predictionForGivenNumDaysDict.get)
			minAccuracy = (minIndex, predictionForGivenNumDaysDict[minIndex])
			median = np.median(predictionAccuracies)

			numDaysTuple = (numDayIndex, numDayStock)
			predictionDict[numDaysTuple] = {'mean':meanAccuracy, 'max':predictionForGivenNumDaysDict[maxIndex], 'min':predictionForGivenNumDaysDict[minIndex], 'median':median }

	sortedTuples = sorted(predictionDict.keys())
	for numDaysTuple in sortedTuples:
		# print "%s:\t %s\n" % (numDaysTuple, predictionDict[numDaysTuple])
		sumStats = predictionDict[numDaysTuple]
		print "& %d & %d & %f & %f & %f & %f \\\\\n" % (numDaysTuple[0], numDaysTuple[1], sumStats['mean'], sumStats['median'], sumStats['max'], sumStats['min'])

if __name__ == "__main__": 
	main()
