import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

##@ Author....Maninder Singh
#@ Modified by Paul Court for SE512 Homework 4 Problem 4 using data for Problem 1.

# read from csv file
data = r'C:\Users\court\Google Drive\SCSU\SE 512\Assignments\HA 4\problem1.txt'
temp = np.loadtxt(data)

clusterSize = 3
notDone = True

## Creating an array of data vectors.

#print (temp)
rows = temp.shape[0]
cols = temp.shape[1]

#Finds the initial random seed value for k vectors (not used in this test case).

#randSize = 500
#Creates an array large enough for the initial cluster vector and populates with random numbers.

#cluster = np.random.randint(randSize, size = (clusterSize, cols))

#Creates an array large enough for the initial cluster vectors.
cluster = np.empty([rows - 8, cols])

#Creates an augmented array to hold the initial vectors, the distances matrix, the cluster number and a temporary
#that can check to see if vectors' cluster points change from trial to trial
#[x1, x2, x3, ... xn, d1, d2, d3,...dn, cluster assignment for this iteration, assignment from last iteration]
X = np.zeros([rows - 3, cols + clusterSize + 2])

#Function to determine the cluster to which a designated vector belongs and stores in the augmented arrray X
#in the first available column after the elements and the distance matrix.

def findMinimumLocation(X, i, cols, clusterSize):
    positionOfMin = 0
    for n in range (0, clusterSize):
        if (n == 0):
            min = X[i, cols]
            positionOfMin =  1
        else:            
            if (X[i, cols + n] < min):
                min = X[i, cols + n]
                positionOfMin = n + 1
    X[i, cols + clusterSize] = positionOfMin

##Calculates the Euclidean Distance from each cluster vector to the current vector.

def PopulateDistances(cluster, clusterSize, X, cols, k): 
    dist = 0
    for j in range (0, cols):
        dist = dist + (cluster[k, j] - X[i, j])**2
    X[i, k + cols] = "{:.2f}".format(dist**(0.5) )     

#Keeps a running sum of each cluster for updating the cluster points.

def clusterPointUpdate(X, i, cols, clusterSize):
    for k in range(0, clusterSize):
        if (X[i, cols + clusterSize] == k+1):
            count[k] = count[k] + 1
            for j in range (0, cols):
                clusterPoint[k,j] = clusterPoint[k,j] + X[i, j]
                
#Check for complete, comparing cluster points from trial to trial.

def checkComplete(X, i, cols, clusterSize):
    if (X[i, cols + clusterSize + 1] == X[i, cols + clusterSize]):
        notDone = False
    else:
        notDone = True
    return notDone

#Moves the first trial results to a temporary column to prepare for the next iteration.

def swapToTemp(X, rows, cols, clusterSize):
    for i in range (0, rows):
        X[i, cols + clusterSize + 1] =  X[i, cols + clusterSize]

#Updates the value of the cluster vector values using means.

def recalculateClusterPoints(cluster, clusterSize, clusterPoint, count, cols):
    for i in range (0, clusterSize):
        for j in range (0, cols):
            temp1 = clusterPoint[i, j]
            temp2 = count[i]
            cluster[i, j] = float(temp1/temp2)  #Watch out for integer divide.
    
##Enters information into the original information (temp) into an augmented array (X).

for i in range (0, rows - 3):
    for j in range (0, cols):
        X[i, j] = temp[i, j]
        
for i in range (8, rows):
    for j in range (0, cols):
        cluster[i - 8, j] = temp[i, j]
        
#Main program for k-Means Algorithm.
while notDone:
    #Creates an arrays for updating the cluster vector average values and 
    #re-initializes them for the next trial.
    count = np.zeros([clusterSize])
    clusterPoint = np.zeros([clusterSize, cols])
    
    for i in range (0, rows-3): 
        for k in range(0, clusterSize):
            PopulateDistances(cluster, clusterSize, X, cols, k)
            findMinimumLocation(X, i, cols, clusterSize)
            notDone = checkComplete(X, i, cols, clusterSize)
        clusterPointUpdate(X, i, cols, clusterSize)
        #Check to see if the vector causes a change of cluster, if so, continue the process by
        #temporarily storing the cluster assignment into the last column of X.
        if (notDone):
            swapToTemp(X, rows - 3, cols, clusterSize)
            
    recalculateClusterPoints(cluster, clusterSize, clusterPoint, count, cols)
    
    #print (clusterPoint)
    #print(count)
    print('Columns: x, y, d1, d2, d3, cluster, temp')
    print(X)
    print('cluster pts:')
    print (cluster)
    print('More iterations?')
    print(notDone)

#Places the six orginal parameters of the vector in an array columns 0 - 5, with 6 beinging the cluster assignment.
graphData = np.zeros([rows-3, cols + 1])
for i in range (0, rows-3):
    for j in range (0, cols):
        graphData[i, j] = X[i, j]
        #Fills teh graphData array with the vector values and the cluster index
        graphData[i, j + 1] = X[i, j + clusterSize + 1] 

df1 = pd.DataFrame(graphData)
sns.color_palette("mako", as_cmap=True)
sns.pairplot(df1, hue= 2)
plt.show()


