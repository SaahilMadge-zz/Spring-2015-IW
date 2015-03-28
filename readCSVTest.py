import csv

ifile = open('testIBMMSFTAAPL.csv', 'rb')
reader = csv.reader(ifile)

rownum = 0
msftCount = 0
ibmCount = 0
aaplCount = 0

for row in reader:
    # Save header row
    if rownum == 0:
        header = row 
    #print row
    if "MSFT" in row:
        msftCount += 1
    elif "IBM" in row:
        ibmCount += 1
    elif "AAPL" in row:
        aaplCount += 1
    else:
        if rownum > 0:
            print "messed up row %s", row

    rownum += 1

ifile.close()

print "MSFT: %d\t IBM: %d\t AAPL: %d\t" % (msftCount, ibmCount, aaplCount)
