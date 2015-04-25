import pandas
import numpy as np
from sklearn import preprocessing
from sklearn import svm

# read the data
df = pandas.read_csv('techsectordatareal.csv')

# calculate price volatility array given company
def calcPriceVolatility(numDays, priceArray):
	# make price volatility array
	volatilityArray = []
	movingVolatilityArray = []
	for i in range(1, numDays+1):
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
	volatilityArray.append(np.mean(movingVolatilityArray))
	for i in range(numDays + 1, len(priceArray) - 1):
		del movingVolatilityArray[0]
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
		volatilityArray.append(np.mean(movingVolatilityArray))
	return volatilityArray

# calculate momentum array
def calcMomentum(numDays, priceArray):
	# now calculate momentum
	momentumArray = []
	movingMomentumArray = []
	for i in range(1, numDays + 1):
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
	momentumArray.append(np.mean(movingMomentumArray))
	for i in range(numDays+1, len(priceArray) - 1):
		del movingMomentumArray[0]
		movingMomentumArray.append(1 if priceArray[i] > priceArray[i-1] else -1)
		momentumArray.append(np.mean(movingMomentumArray))
	return momentumArray

def makeModelAndPredict(permno): 
	companyData = df[df['PERMNO'] == permno]
	companyPrices = list(companyData['PRC'])
	# print len(companyPrices)

	# get up to 2011
	datesArray = list(companyData['date'])
	# datesArray = list(datesArray.astype('str'))
	startOfTwelve = datesArray.index(20120103)

	numDays = 1 # how many days to use for moving averages
	startNum = 5  # to allow for 5 days of momentum at the start

	# basic feature set: average of last 5 prices, momentum (average of last 5 ups or downs), and market cap
	volatilityArray = calcPriceVolatility(numDays,companyPrices)
	momentumArray = calcMomentum(numDays,companyPrices)

	splitIndex = startOfTwelve - numDays

	# make the feature array
	X = []
	X.append(volatilityArray[:splitIndex])
	X.append(momentumArray[:splitIndex])
	X = np.array(X)

	X_scaled = []
	X_scaled.append(preprocessing.scale(X[0]))
	X_scaled.append(preprocessing.scale(X[1]))
	X_scaled = np.transpose(np.array(X_scaled))

	# make the output array for training
	Y = []
	for i in range(numDays, len(companyPrices) - 1):
		Y.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
	Y_training = Y[:splitIndex]
	# print Y

	# create the SVM model
	rbf_svc = svm.SVC(kernel = 'rbf')
	print rbf_svc.fit(X_scaled, Y_training)

	# predict on the test data

	# X_test = np.array([priceAverageArray[splitIndex:], momentumArray[splitIndex:], companyShares[splitIndex:]])
	X_test = []
	X_test.append(volatilityArray[splitIndex:])
	X_test.append(momentumArray[splitIndex:])
	X_test = np.array(X_test)

	X_test_scaled = []
	X_test_scaled.append(preprocessing.scale(X_test[0]))
	X_test_scaled.append(preprocessing.scale(X_test[1]))
	X_test_scaled = np.transpose(np.array(X_test_scaled))

	predictions = rbf_svc.predict(X_test_scaled)

	# test predictions
	Y_test = Y[splitIndex:]

	truthArray = []
	for i in range(len(Y_test)):
		# print "prediction: %d\tactual: %d" % (predictions[i], Y_test[i])
		truthArray.append(1. if predictions[i] == Y_test[i] else 0.)

	# print len(truthArray)
	print sum(truthArray)
	# print np.array(truthArray)

	return sum(truthArray)/len(truthArray)

def main():
	global df
	permnoList = sorted(set(list(df['PERMNO'])))
	companiesNotFull = [12084, 13407, 14542, 93002] # companies without full dates

	predictionForGivenNumDaysDict = {}

	for permno in permnoList:
		if permno in companiesNotFull:
			continue
		print permno
		predictionForGivenNumDaysDict[permno] = makeModelAndPredict(permno)

	print predictionForGivenNumDaysDict

if __name__ == '__main__':
	main()