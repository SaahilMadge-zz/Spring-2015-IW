import pandas
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation

# read the data
df = pandas.read_csv('techsectordatareal.csv')

# calculate price volatility array given company
def calcPriceVolatility(numDays, priceArray):
	# make price volatility array
	volatilityArray = []
	movingVolatilityArray = []
	for i in range(1, numDays+1):
		# print priceArray[i]
		# print priceArray[i-1]
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		movingVolatilityArray.append(percentChange)
	volatilityArray.append(np.mean(movingVolatilityArray))
	for i in range(numDays + 1, len(priceArray) - 1):
		del movingVolatilityArray[0]
		percentChange = 100 * (priceArray[i] - priceArray[i-1]) / priceArray[i-1]
		if np.isnan(priceArray[i]):
			print 'HERE'
			print i
		if np.isnan(priceArray[i]):
			print 'HERE!'
			print i
		movingVolatilityArray.append(percentChange)
		volatilityArray.append(np.mean(movingVolatilityArray))
		if np.isnan(np.mean(movingVolatilityArray)):
			print 'HERE!'
			print i
	print len(volatilityArray)
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
	print len(momentumArray)
	return momentumArray

def makeModelAndPredict(permno, numDays, sectorVolatility, sectorMomentum, splitNumber):
	global df
	# get price volatility and momentum for this company
	companyData = df[df['PERMNO'] == permno]
	companyPrices = list(companyData['PRC'])
	# print companyPrices
	volatilityArray = calcPriceVolatility(numDays, companyPrices)
	# print momentumArray
	# print np.isnan(volatilityArray[1468])
	momentumArray = calcMomentum(numDays, companyPrices)
	# print momentumArray

	splitIndex = splitNumber - numDays

# 	# fit model
# 	X = []
# 	X.append(volatilityArray[splitIndex:])
# 	X.append(momentumArray[splitIndex:])
# 	X.append(sectorVolatility[splitIndex:])
# 	X.append(sectorMomentum[splitIndex:])
# 	X = np.array(X)

# 	X_scaled = []
# 	X_scaled.append(preprocessing.scale(X[0]))
# 	X_scaled.append(preprocessing.scale(X[1]))
# 	X_scaled.append(preprocessing.scale(X[2]))
# 	X_scaled.append(preprocessing.scale(X[3]))
# 	X_scaled = np.transpose(np.array(X_scaled))

# 	# make the output array for training
# 	Y = []
# 	for i in range(numDays, len(companyPrices) - 1):
# 		Y.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
# 	Y_training = Y[splitIndex:]

# 	# create the SVM model
# 	rbf_svc = svm.SVC(kernel = 'rbf')
# 	print rbf_svc.fit(X_scaled, Y_training)

# 	# predict on the test data

# 	X_test = []
# 	X_test.append(volatilityArray[:splitIndex])
# 	X_test.append(momentumArray[:splitIndex])
# 	X_test.append(sectorVolatility[:splitIndex])
# 	X_test.append(sectorMomentum[:splitIndex])
# 	X_test = np.array(X_test)
# # 
# 	X_test_scaled = []
# 	X_test_scaled.append(preprocessing.scale(X_test[0]))
# 	X_test_scaled.append(preprocessing.scale(X_test[1]))
# 	X_test_scaled.append(preprocessing.scale(X_test[2]))
# 	X_test_scaled.append(preprocessing.scale(X_test[3]))
# 	X_test_scaled = np.transpose(np.array(X_test_scaled))

# 	predictions = rbf_svc.predict(X_test_scaled)
# 	print predictions

# 	# test predictions
# 	Y_test = Y[:splitIndex]
# 	print Y_test
# 	print len(Y_test)
# 	print len(predictions)
# 	print ""

# 	truthArray = []
# 	for i in range(len(Y_test)):
# 		truthArray.append(1. if predictions[i] == Y_test[i] else 0.)

# 	print sum(truthArray)
# 	print rbf_svc.score(X_test_scaled, Y_test)
# 	# return sum(truthArray)/len(truthArray)

	X = np.transpose(np.array([volatilityArray, momentumArray, sectorVolatility, sectorMomentum]))
	# X = np.arange(len(volatilityArray))
	# print np.shape(X) 
	# print X
	Y = []
	for i in range(numDays, len(companyPrices) - 1):
		Y.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
	X_train = np.array(X[0:3]).astype('float64').copy(order='C')
	X_test = np.array(X[3:]).astype('float64').copy(order='C')
	y_train = np.array(Y[0:3]).astype('float64').copy(order='C')
	y_test = np.array(Y[3:]).astype('float64').copy(order='C')
	print X_train.shape
	print y_train.shape
	# print np.shape(Y)
	# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size = 0.4, random_state=0)
	# print len(X_train)
	# print len(y_train)
	# print len(X_test)
	# print len(y_test)
	# print X
	# print X_train
	# print y_train
	# print X_test
	# print y_test
	rbf_svm = svm.SVC(kernel='rbf')
	rbf_svm.fit(X_train, y_train)
	# print svm.libsvm.predict(X_test)
	print rbf_svm.score(X_test, y_test)

def main():
	global df
	permnoList = sorted(set(list(df['PERMNO'])))
	# for i in permnoList:
	# 	print i
	print len(permnoList) # should be 39
	print permnoList

	# for permno in permnoList:
	# 	print "%d:\t%d" % (permno, list(df[df['PERMNO'] == permno]['date']).__contains__(20070103))

	companiesNotFull = [12084, 13407, 14542, 93002] # companies without full dates

	# read the tech sector data
	ndxtdf = pandas.read_csv('ndxtdata.csv')
	ndxtdf = ndxtdf.sort_index(by='Date', ascending=True)
	ndxtPrices = list(ndxtdf['Close'])
	# print ndxtdf
	# ndxtDates = ndxtdf['Date']
	# ndxtDateInts = []
	# print ndxtDates
	# print len(ndxtDates)
	# print len(df[df['PERMNO'] == 10107])

	# find when 2012 starts
	startOfTwelve = list(df[df['PERMNO'] == 10107]['date']).index(20120103)

	# print df[df['PERMNO'] == 10107]['date']

	# for date in ndxtDates:
	# 	dateString = str(date[0:4])
	# 	dateString += str(date[5:7])
	# 	dateString += str(date[8:10])
	# 	ndxtDateInts.append(int(dateString))

	# msftDatesList = list(df[df['PERMNO'] == 10107]['date'])

	# for date in msftDatesList:
	# 	if ndxtDateInts.__contains__(date):
	# 		continue
	# 	print "%s:\t%d" % (date, ndxtDateInts.__contains__(date))


	# WE NOTICE THAT THESE DAYS ARE MISSING FROM NDXT: 20070226   20081027  20100714 so we add them in with 0 price change


	# we want to predict where it will be on the next day based on X days previous
	numDaysArray = [1, 5, 20, 90, 270] # day, week, month, quarter, year

	predictionDict = {}

	# for numDay in numDaysArray:
	ndxtVolatilityArray = calcPriceVolatility(numDaysArray[1], ndxtPrices)
	ndxtMomentumArray = calcMomentum(numDaysArray[1], ndxtPrices)
	predictionForGivenNumDaysDict = {}

	# for permno in permnoList:
	# 	if permno in companiesNotFull:
	# 		continue
	# 	print permno
	# 	percentage = makeModelAndPredict(permno,numDaysArray[3],ndxtVolatilityArray,ndxtMomentumArray,startOfTwelve)
	# 	predictionForGivenNumDaysDict[permno] = percentage

	percentage = makeModelAndPredict(10909, numDaysArray[1],ndxtVolatilityArray,ndxtMomentumArray,startOfTwelve)
	# print percentage

	# print predictionForGivenNumDaysDict

		# predictionDict[numDay] = predictionForGivenNumDaysDict

if __name__ == "__main__": 
	main()
