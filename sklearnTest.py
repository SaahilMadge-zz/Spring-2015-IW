# Test for sklearn with Microsoft stock, using years 2007-2011 as training and 2012-2014 as test data

# Read the data in
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm

# we will be using -1 for a down day, +1 for an up day

df = pandas.read_csv('sp500data2.csv')

def makeModelAndPredict(permno): 
	companyData = df[df['PERMNO'] == permno]
	companyPrices = list(companyData['PRC'])
	# print len(companyPrices)

	# get up to 2011
	datesArray = np.array(list(companyData['date']))
	datesArray = list(datesArray.astype('str'))
	startOfTwelve = datesArray.index('20120103')

	startNum = 5  # to allow for 5 days of momentum at the start


	# basic feature set: average of last 5 prices, momentum (average of last 5 ups or downs), and market cap
	priceAverageArray = []
	for i in range(startNum, len(companyPrices) - 1):
		priceAverageArray.append(np.mean(companyPrices[i-4:i]))

	# print priceAverageArray

	# now calculate momentum
	momentumArray = []
	movingAverageArray = []
	movingAverageArray.append(1 if companyPrices[1] > companyPrices[0] else -1)
	movingAverageArray.append(1 if companyPrices[2] > companyPrices[1] else -1)
	movingAverageArray.append(1 if companyPrices[3] > companyPrices[2] else -1)
	movingAverageArray.append(1 if companyPrices[4] > companyPrices[3] else -1)
	movingAverageArray.append(1 if companyPrices[5] > companyPrices[4] else -1)
	momentumArray.append(np.mean(movingAverageArray))
	for i in range(6, len(companyPrices) - 1):
		del movingAverageArray[0]
		movingAverageArray.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
		momentumArray.append(np.mean(movingAverageArray))

	# print momentumArray

	# shares outstanding
	companyShares = np.array(list(companyData['SHROUT']))
	# microsoftShares = list(microsoftShares.astype(int))
	companyShares = companyShares[startNum:-1]


	# print startOfTwelve
	splitIndex = startOfTwelve - startNum
	# print len(priceAverageArray[:splitIndex])
	# print len(momentumArray[:splitIndex])
	# print len(companyShares[:splitIndex])

	# make the feature array
	X = np.array([priceAverageArray[:splitIndex], momentumArray[:splitIndex], companyShares[:splitIndex]])
	# X = np.array(priceAverageArray[4:])
	# print X

	# X_scaled = preprocessing.scale(X)

	# scale the features
	X_scaled = np.transpose(np.array([preprocessing.scale(X[0]), preprocessing.scale(X[1]), preprocessing.scale(X[2])]))
	# print X_scaled
	# print len(X_scaled)
	# print X_scaled[0].mean()
	# print X_scaled[0].std()
	# print X_scaled[1].mean()
	# print X_scaled[1].std()
	# print X_scaled[2].mean()
	# print X_scaled[2].std()

	# make the output array for training
	Y = []
	for i in range(5, len(companyPrices) - 1):
		Y.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
	Y_training = Y[:splitIndex]
	# print Y

	# create the SVM model
	rbf_svc = svm.SVC(kernel = 'rbf')
	print rbf_svc.fit(X_scaled, Y_training)

	# predict on the test data

	X_test = np.array([priceAverageArray[splitIndex:], momentumArray[splitIndex:], companyShares[splitIndex:]])
	X_test_scaled = np.transpose(np.array([preprocessing.scale(X_test[0]), preprocessing.scale(X_test[1]), preprocessing.scale(X_test[2])]))

	predictions = rbf_svc.predict(X_test_scaled)
	# print type(predictions)

	# test predictions
	Y_test = Y[splitIndex:]

	# print len(X_test_scaled)
	# print len(predictions)
	# print len(Y_test)

	truthArray = []
	for i in range(len(Y_test)):
		# print "prediction: %d\tactual: %d" % (predictions[i], Y_test[i])
		truthArray.append(1. if predictions[i] == Y_test[i] else 0.)

	# print len(truthArray)
	print sum(truthArray)
	# print np.array(truthArray)

	return sum(truthArray)/len(truthArray)


predictionDict = {}
microsoftPermno = 10107
# predictionDict[microsoftPermno] = int(makeModelAndPredict(microsoftPermno) * 100)
predictionDict[microsoftPermno] = makeModelAndPredict(microsoftPermno)

cocaColaPermno = 11308
# predictionDict[cocaColaPermno] = int(makeModelAndPredict(cocaColaPermno) * 100)
predictionDict[cocaColaPermno] = makeModelAndPredict(cocaColaPermno)

xomPermno = 11850
predictionDict[xomPermno] = makeModelAndPredict(xomPermno)

gePermno = 12060
predictionDict[gePermno] = makeModelAndPredict(gePermno)

cvxPermno = 14541
predictionDict[cvxPermno] = makeModelAndPredict(cvxPermno)

aaplPermno = 14593
predictionDict[aaplPermno] = makeModelAndPredict(aaplPermno)

radioshackPermno = 15560
predictionDict[radioshackPermno] = makeModelAndPredict(radioshackPermno)

print predictionDict
