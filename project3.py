import matplotlib.pyplot as plot
import csv
import random
import math
import numpy as np

#makes a prediction for gradient descent given the current coefficient and x value
def makePrediction(coeff, var):
    prediction = 0
    for i in range(len(coeff)):
        prediction = prediction + coeff[i]*var**i
    return prediction

#generated 1st degree polynomial regression 
def firstDegree(coeff, trainData):
    stopMark = 10**-3
    oldCoeff = [0,0]
    coeffDiff = [0,0]
    stop = False
    learnRate = 10**-3
    while stop == False:
        #make predictions
        predictions = []
        for i in range(len(trainData)):
            predictions.append(makePrediction(coeff, trainData[i][0]))
        #adjust polynomial coefficients
        for i in range(len(coeff)):
            sumDiff = 0
            for j in range(len(predictions)):
                sumDiff = sumDiff + (predictions[j] - trainData[j][1]) * trainData[j][0]**i
            coeff[i] = coeff[i] - learnRate * (1 / len(trainData)) * sumDiff
        #determines if every element in the differences of coefficients meet the stopping criteria
        for i in range(len(coeff)):
            coeffDiff[i] = abs(coeff[i] - oldCoeff[i]) 
        stop = all(x < stopMark for x in coeffDiff)
        oldCoeff = coeff.copy()

#generated 2nd degree polynomial regression 
def secondDegree(coeff, trainData):
    stopMark = 10**-3
    oldCoeff = [0,0,0]
    coeffDiff = [0,0,0]
    stop = False
    learnRate = 10**-3
    while stop == False:
        #make predictions
        predictions = []
        for i in range(len(trainData)):
            predictions.append(makePrediction(coeff, trainData[i][0]))
        #adjust polynomial coefficients
        for i in range(len(coeff)):
            sumDiff = 0
            for j in range(len(predictions)):
                sumDiff = sumDiff + (predictions[j] - trainData[j][1]) * trainData[j][0]**i
            coeff[i] = coeff[i] - learnRate * (1 / len(trainData)) * sumDiff
        #determines if every element in the differences of coefficients meet the stopping criteria
        for i in range(len(coeff)):
            coeffDiff[i] = abs(coeff[i] - oldCoeff[i]) 
        stop = all(x < stopMark for x in coeffDiff)
        oldCoeff = coeff.copy()

#generated 4th degree polynomial regression        
def fourthDegree(coeff, trainData):
    stopMark = 10**-3
    oldCoeff = [0,0,0,0,0]
    coeffDiff = [0,0,0,0,0]
    stop = False
    learnRate = 10**-3
    while stop == False:
        #make predictions
        predictions = []
        for i in range(len(trainData)):
            predictions.append(makePrediction(coeff, trainData[i][0]))
        #adjust polynomial coefficients
        for i in range(len(coeff)):
            sumDiff = 0
            for j in range(len(predictions)):
                sumDiff = sumDiff + (predictions[j] - trainData[j][1]) * trainData[j][0]**i
            coeff[i] = coeff[i] - learnRate * (1 / len(trainData)) * sumDiff
        #determines if every element in the differences of coefficients meet the stopping criteria
        for i in range(len(coeff)):
            coeffDiff[i] = abs(coeff[i] - oldCoeff[i]) 
        stop = all(x < stopMark for x in coeffDiff)
        oldCoeff = coeff.copy()

#generated 7th degree polynomial regression         
def seventhDegree(coeff, trainData):
    stopMark = 10**-3
    oldCoeff = [0,0,0,0,0,0,0,0]
    coeffDiff = [0,0,0,0,0,0,0,0]
    stop = False
    learnRate = 10**-3
    while stop == False:
        #make predictions
        predictions = []
        for i in range(len(trainData)):
            predictions.append(makePrediction(coeff, trainData[i][0]))
        #adjust polynomial coefficients
        for i in range(len(coeff)):
            sumDiff = 0
            for j in range(len(predictions)):
                sumDiff = sumDiff + (predictions[j] - trainData[j][1]) * trainData[j][0]**i
            coeff[i] = coeff[i] - learnRate * (1 / len(trainData)) * sumDiff
        #determines if every element in the differences of coefficients meet the stopping criteria
        for i in range(len(coeff)):
            coeffDiff[i] = abs(coeff[i] - oldCoeff[i]) 
        stop = all(x < stopMark for x in coeffDiff)
        oldCoeff = coeff.copy()

#calculates the mean squared error for a given polynomial regression
def calcError():
    allErrors = []
    for i in range(len(testSets)):#for each data set
        m = len(testSets[i])
        sumDiff = 0
        testErrors = []
        for j in range(len(allCoeff[i])):#for each degree polynomial calculate error
            for k in range(m):#for each test case calculate the summation for error equation
                sumDiff = sumDiff + (makePrediction(allCoeff[i][j], testSets[i][k][0]) - testSets[i][k][1]) ** 2
            error = (1 / m) * sumDiff
            testErrors.append(error)
        allErrors.append(testErrors)
    return allErrors

#generates the string form of the generated polynomial regressions and returns them in a 2D list
def writePolynomials():
    allStrings = []
    for i in range(len(allCoeff)):#for each data set
        polyStrings = []
        for j in range(len(allCoeff[i])):#for each degree polynomial
            string = ''
            for k in range(len(allCoeff[i][j])):#for each polynomial term add the coefficient and variable to the string
                if (k != len(allCoeff[i][j]) - 1):
                    string = string + str('%.3f'%allCoeff[i][j][k]) + 'x^' + str(k) + ' + '
                else:
                    string = string + str('%.3f'%allCoeff[i][j][k]) + 'x^' + str(k)
            polyStrings.append(string)
        allStrings.append(polyStrings)
    return allStrings

#calculates the polynomial figure to be plotted on the scatterplots            
def calcPoly(x, coeff):
    y = 0
    for i in range(len(coeff)):
        y = y + coeff[i]*x**i
    return y

#outputs the graphs with training points and generated models
def printGraphs(): 
    for i in range(len(trainSets)):#for each dataset
        plotDataX = []
        plotDataY = []
        for j in range(len(train1)):#for each degree polynomial
           plotDataX.append(trainSets[i][j][0])
           plotDataY.append(trainSets[i][j][1])
        plot.scatter(plotDataX, plotDataY)
        plot.xlim(math.floor(min(plotDataX)), math.ceil(max(plotDataX)))
        plot.ylim(math.floor(min(plotDataY)), math.ceil(max(plotDataY)))
        x = np.linspace(min(plotDataX), max(plotDataX))
        degrees = [1, 2, 4, 7]
        for j in range(len(allCoeff[0])):
            plot.plot(x, calcPoly(x, allCoeff[i][j]), label = 'Degree ' + str(degrees[j]))
        plot.legend(loc='center', bbox_to_anchor=(0.5, -0.20), ncol=4)
        plot.title(data[i])
        plot.xlabel('x')
        plot.ylabel('y')
        plot.show()

#prints the text output after the graphs are printed out
def printText():
    print("--------------------------------------")
    for i in range(len(polyStrings)):
        print("Dataset " + str(i+1) + ":")
        for j in range(len(polyStrings[i])):
            print("Polynomial: " + polyStrings[i][j])
            print("Error: " + str(errorList[i][j]))
            print()
        print("--------------------------------------")

#import the csv data into 2 dimensional lists
synData1 = []
synData2 = []
synData3 = []
data = ['synthetic-1.csv', 'synthetic-2.csv', 'synthetic-3.csv']

#insert the imported data into lists
for i in data:
    with open(i) as csvFile:
        readFile = csv.reader(csvFile, delimiter = ',')
        for row in readFile:
            insertList = [float(row[0]), float(row[1])]
            if i == data[0]:
                synData1.append(insertList)
            elif i == data[1]:
                synData2.append(insertList)
            else:
                synData3.append(insertList)

#randomly assign the loaded data into training and testinf sets at a ratio of 70% training and 30% testing
random.shuffle(synData1)
random.shuffle(synData2)
random.shuffle(synData3)
train1 = synData1[30:]
test1 = synData1[70:]
train2 = synData2[30:]
test2 = synData2[70:]
train3 = synData3[30:]
test3 = synData3[70:]

trainSets = [train1, train2, train3]
testSets = [test1, test2, test3]

#store the coefficients for every degree polynomial of each dataset
allCoeff = []

#calculate the coefficients for each polynomial
for i in range(len(trainSets)):
    degOneCoeff = []
    degTwoCoeff = []
    degFourCoeff = []
    degSevenCoeff = []
    for i2 in range(2):
        degOneCoeff.append(uniform(1, 10))
    for i3 in range(3):
        degTwoCoeff.append(uniform(1, 10))
    for i4 in range(5):
        degFourCoeff.append(uniform(1, 10))
    for i5 in range(8):
        degSevenCoeff.append(uniform(1, 10))
    firstDegree(degOneCoeff, trainSets[i])
    secondDegree(degTwoCoeff, trainSets[i])
    fourthDegree(degFourCoeff, trainSets[i])
    seventhDegree(degSevenCoeff, trainSets[i])
    partialCoeff = [degOneCoeff, degTwoCoeff, degFourCoeff, degSevenCoeff]
    allCoeff.append(partialCoeff)

#find and store the model errors
errorList = calcError()

#find and store the polynomial strings for output
polyStrings = writePolynomials()

#print the scatterplots with models for the user
printGraphs()

#print the polynomial strings and errors for the user
printText()