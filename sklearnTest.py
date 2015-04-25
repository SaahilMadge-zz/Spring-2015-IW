# Test for sklearn with Microsoft stock, using years 2007-2011 as training and 2012-2014 as test data

# Read the data in
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
import operator

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

	numDays = 5 # how many days to use for moving averages
	startNum = 5  # to allow for 5 days of momentum at the start

	# basic feature set: average of last 5 prices, momentum (average of last 5 ups or downs), and market cap

	# make price volatility array
	volatilityArray = []
	movingVolatilityArray = []
	for i in range(1, numDays+1):
		percentChange = 100 * (companyPrices[i] - companyPrices[i-1]) / companyPrices[i-1]
		movingVolatilityArray.append(percentChange)
	volatilityArray.append(np.mean(movingVolatilityArray))
	for i in range(numDays + 1, len(companyPrices) - 1):
		del movingVolatilityArray[0]
		percentChange = 100 * (companyPrices[i] - companyPrices[i-1]) / companyPrices[i-1]
		movingVolatilityArray.append(percentChange)
		volatilityArray.append(np.mean(movingVolatilityArray))

	# print priceAverageArray

	# now calculate momentum
	momentumArray = []
	movingMomentumArray = []
	for i in range(1, numDays + 1):
		movingMomentumArray.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
	momentumArray.append(np.mean(movingMomentumArray))

	# movingAverageArray.append(1 if companyPrices[1] > companyPrices[0] else -1)
	# movingAverageArray.append(1 if companyPrices[2] > companyPrices[1] else -1)
	# movingAverageArray.append(1 if companyPrices[3] > companyPrices[2] else -1)
	# movingAverageArray.append(1 if companyPrices[4] > companyPrices[3] else -1)
	# movingAverageArray.append(1 if companyPrices[5] > companyPrices[4] else -1)
	# momentumArray.append(np.mean(movingAverageArray))
	for i in range(numDays+1, len(companyPrices) - 1):
		del movingMomentumArray[0]
		movingMomentumArray.append(1 if companyPrices[i] > companyPrices[i-1] else -1)
		momentumArray.append(np.mean(movingMomentumArray))

	# print momentumArray

	# shares outstanding
	companyShares = np.array(list(companyData['SHROUT']))
	# microsoftShares = list(microsoftShares.astype(int))
	companyShares = companyShares[numDays:-1]


	# print startOfTwelve
	splitIndex = startOfTwelve - numDays
	# print len(priceAverageArray[:splitIndex])
	# print len(momentumArray[:splitIndex])
	# print len(companyShares[:splitIndex])

	# make the feature array
	# X = np.array([priceAverageArray[:splitIndex], momentumArray[:splitIndex], companyShares[:splitIndex]])
	X = []
	X.append(volatilityArray[:splitIndex])
	X.append(momentumArray[:splitIndex])
	X.append(companyShares[:splitIndex])
	X = np.array(X)

	# X = np.array(priceAverageArray[4:])
	# print X

	# X_scaled = preprocessing.scale(X)

	# scale the features
	# X_scaled = np.transpose(np.array([preprocessing.scale(X[0]), preprocessing.scale(X[1]), preprocessing.scale(X[2])]))

	X_scaled = []
	X_scaled.append(preprocessing.scale(X[0]))
	X_scaled.append(preprocessing.scale(X[1]))
	# X_scaled.append(preprocessing.scale(X[2]))
	X_scaled = np.transpose(np.array(X_scaled))

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

	# X_test = np.array([priceAverageArray[splitIndex:], momentumArray[splitIndex:], companyShares[splitIndex:]])
	X_test = []
	X_test.append(volatilityArray[splitIndex:])
	X_test.append(momentumArray[splitIndex:])
	X_test.append(companyShares[splitIndex:])
	X_test = np.array(X_test)

	# X_test_scaled = np.transpose(np.array([preprocessing.scale(X_test[0]), preprocessing.scale(X_test[1]), preprocessing.scale(X_test[2])]))
	X_test_scaled = []
	X_test_scaled.append(preprocessing.scale(X_test[0]))
	X_test_scaled.append(preprocessing.scale(X_test[1]))
	# X_test_scaled.append(preprocessing.scale(X_test[2]))
	X_test_scaled = np.transpose(np.array(X_test_scaled))

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
# microsoftPermno = 10107
# # predictionDict[microsoftPermno] = int(makeModelAndPredict(microsoftPermno) * 100)
# predictionDict[microsoftPermno] = makeModelAndPredict(microsoftPermno)

# cocaColaPermno = 11308
# # predictionDict[cocaColaPermno] = int(makeModelAndPredict(cocaColaPermno) * 100)
# predictionDict[cocaColaPermno] = makeModelAndPredict(cocaColaPermno)

# xomPermno = 11850
# predictionDict[xomPermno] = makeModelAndPredict(xomPermno)

# gePermno = 12060
# predictionDict[gePermno] = makeModelAndPredict(gePermno)

# cvxPermno = 14541
# predictionDict[cvxPermno] = makeModelAndPredict(cvxPermno)

# aaplPermno = 14593
# predictionDict[aaplPermno] = makeModelAndPredict(aaplPermno)

# radioshackPermno = 15560
# predictionDict[radioshackPermno] = makeModelAndPredict(radioshackPermno)

# hersheyPermno = 16600
# predictionDict[hersheyPermno] = makeModelAndPredict(hersheyPermno)

permnoList = sorted(list(set(df['PERMNO'])))
for permno in permnoList:
	companyData = df[df['PERMNO'] == permno]
	datesArray = np.array(list(companyData['date']))
	datesArray = list(datesArray.astype('str'))

	if ((not datesArray.__contains__('20070103')) or (not datesArray.__contains__('20141231'))):
		continue
	if permno == 11896 or permno == 15579 or permno == 38762 or permno == 56274 or permno == 69032 or permno == 76744:   # skip MAXIM INTEGRATED and TEXAS INSTRUMENTS and NISOURCE INC and CONAGRA INC and MORGAN STANLEY and VERTEX PHARMA
		continue
	print permno
	try:
		predictionValue = makeModelAndPredict(permno)
		predictionDict[permno] = predictionValue
	except:
		continue


print sorted(predictionDict.items(), key=operator.itemgetter(1))
print "len:%d" % len(predictionDict)
print max(predictionDict.values())
print min(predictionDict.values())

# graph a bar graph
plt.figure(1)
x_pos = np.arange(len(predictionDict.keys()))
plt.bar(x_pos, predictionDict.values())
plt.xticks(x_pos, predictionDict.keys())
plt.ylim([min(predictionDict.values()) - 0.02, max(predictionDict.values()) + 0.02])

plt.show()

