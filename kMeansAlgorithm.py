import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.cluster as cl

##@Author....Maninder Singh
#@ modified by Paul Court for SE512 Homework 4 Problem 4.

## You can use the following library for clustering
from sklearn.cluster import KMeans

## read from csv file
data = pd.read_csv(r'C:\Users\court\Google Drive\SCSU\SE 512\Assignments\HA 4\buddymove.csv')
##Extracting each column
f1= data['Sports']
f2 = data['Religious']
f3 = data['Nature']
f4 = data['Theatre']
f5 = data['Shopping']
f6 = data['Picnic']

## Creating an array of data points
temp = np.array(list(zip(f1,f2,f3,f4,f5,f6)))
rows = temp.shape[0]
cols = temp.shape[1]
clusterSize = 5

#Establishes default values for the random integer and the number of clusters.
#print('Please enter a value for k, the number of clusters: ', clusterSize)
randSize = 200

#Finds the initial random seed value for k vectors.
cluster = np.random.randint(randSize, size = (clusterSize, cols))

#Initializes the recursion key.
notDone = True

#Creates an augmented array to hold the initial vectors, the distances matrix, the cluster number and a temporary
#that can check to see if vectors' cluster points change from trial to trial
#[x1, x2, x3, ... xn, d1, d2, d3,...dn, cluster assignment for this iteration, assignment from last iteration]
#Also, creates an augmented matrix for the original vector and the cluster assignment.  This is most likely
#not the most efficient process for doing this....

X = np.zeros([rows, cols + clusterSize + 2])
graphData = np.zeros([rows, cols + 1])

#Creates two array to hold information on updated cluster points.
clusterPoint = np.zeros([clusterSize, cols])
count = np.zeros([clusterSize])

#Copies the original data into the work array X and the graphing array.
for i in range (0, rows):
    for j in range (0, cols):
        X[i, j] = temp[i, j]
        graphData[i, j] = temp[i, j]
        
#***************************************************************
##Program Function Definitions.
#***************************************************************
##Calculates the Euclidean Distance from each cluster vector to the current vector.
def PopulateDistances(cluster, clusterSize, X, cols, k): 
    dist = 0
    for j in range (0, cols):
        dist = dist + (cluster[k, j] - X[i, j])**2
    X[i, k + cols] = "{:.2f}".format(dist**(0.5))     
    
#***************************************************************
##Function to determine the cluster to which a designated vector belongs and stores in the augmented arrray X
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

#***************************************************************
##Check for complete, comparing cluster points from trial to trial.
def checkComplete(X, i, cols, clusterSize):
    if (X[i, cols + clusterSize + 1] == X[i, cols + clusterSize]):
        notDone = False
    else:
        notDone = True
    return notDone

#***************************************************************
##Keeps a running sum of each cluster for updating the cluster points.
def clusterPointUpdate(X, i, cols, clusterSize):
    for k in range(0, clusterSize):
        if (X[i, cols + clusterSize] == k + 1):
            count[k] = count[k] + 1
            for j in range (0, cols):
                clusterPoint[k,j] = clusterPoint[k,j] + X[i, j]

#***************************************************************
#Moves the current trial results to a temporary column to prepare for the next iteration.
def swapToTemp(X, rows, cols, clusterSize):
    for i in range (0, rows):
        X[i, cols + clusterSize + 1] =  X[i, cols + clusterSize]

#***************************************************************
#Updates the value of the cluster vector values using means.
def recalculateClusterPoints(cluster, clusterSize, clusterPoint, count, cols):
    for i in range (0, clusterSize):
        for j in range (0, cols):
            temp1 = clusterPoint[i, j]
            temp2 = count[i]
            if (temp2 != 0):
                cluster[i, j] = float(temp1/temp2) #watch out for integer divide and divide by zero!
            else:
                cluster[i, j] = clusterPoint[i, j]
 
#***************************************************************
#Main program for k-Means Algorithm.
#***************************************************************
while notDone:
    #Creates an arrays for updating the cluster vector average values and 
    #re-initializes them for the next trial.
    count = np.zeros([clusterSize])
    clusterPoint = np.zeros([clusterSize, cols])

    for i in range (0, rows): 
        for k in range(0, clusterSize):
            PopulateDistances(cluster, clusterSize, X, cols, k)
            findMinimumLocation(X, i, cols, clusterSize)
            notDone = checkComplete(X, i, cols, clusterSize)
        clusterPointUpdate(X, i, cols, clusterSize)
        #Check to see if the vector causes a change of cluster, if so, continue the process by
        #temporarily storing the cluster assignment into the last column of X.
        if (notDone):
            swapToTemp(X, rows, cols, clusterSize)
        #Append the cluster index assigned to the graphData
        graphData[i, cols] = X[i, cols + clusterSize + 1]
    recalculateClusterPoints(cluster, clusterSize, clusterPoint, count, cols)

#***************************************************************
#Print Program Output.
#***************************************************************
print('Original Vectors; Distance Matrix; Cluster Index (Repeated twice)
print('Columns: x1, x2,... xn, d1, d2, .... dn, cluster, temp')
print(X)
print('Cluster Size = ', clusterSize)
print('Cluster Vectors:')
print (cluster)

#***************************************************************
#Outputs visualiazation graphic.
#***************************************************************
df1 = pd.DataFrame(data=graphData)
sns.color_palette("mako", as_cmap=True)
sns.pairplot(df1, hue= 6)
plt.show()

