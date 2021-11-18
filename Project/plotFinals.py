import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np


df = pd.read_csv('finalResults.csv')
#'MeanValue','BestValue','Iteration','TimeLeft'

bestCost = df['BestValue']
meanCost = df['MeanValue']
timeLeft = df['TimeLeft']
timePassed = - timeLeft + 300.0
iterations = df['Iteration']

fig, ax = plt.subplots()
ax.scatter(timePassed, bestCost,label = "Best Path Cost")
ax.scatter(timePassed, meanCost,label = "Mean Cost")
z = np.polyfit(timePassed, bestCost, 1)
p = np.poly1d(z)
ax.plot(timePassed,p(timePassed),label = "Linear Regression of the Best Paths")

ax.set_xlabel('TimePassed (seconds)')  # Add an x-label to the axes.
ax.set_ylabel('Cost of the Path (-)')  # Add a y-label to the axes.
ax.set_title('Result of 100 executions')


dx = 3; xmin = np.min(timePassed); xmax = np.max(timePassed)
dy = 1000; ymin = np.min(bestCost); ymax = np.max(meanCost)

ax.set_xlim([xmin - dx, xmax + dx])
ax.set_ylim([ymin - dy, ymax + dy])
ax.legend()
plt.show()

print("Mean Cost: min max mean variance")
print(ymin,ymax,np.mean(meanCost),np.var(meanCost))
print("best Cost: min max mean variance")
print(np.min(bestCost),np.max(bestCost),np.mean(bestCost),np.var(bestCost))
print("timepassed: min max mean variance")
print(np.min(timePassed),np.max(timePassed),np.mean(timePassed),np.var(timePassed))
print("iterations: min max mean variance")
print(np.min(iterations),np.max(iterations),np.mean(iterations),np.var(iterations))

#Mean Cost: min max mean variance
#27805.24097369539 40002.51489255131 32892.231254832644 8037074.256234057
#best Cost: min max mean variance
#27805.24097369539 39429.6148096179 32322.382711219685 8267312.49599916
#timepassed: min max mean variance
#2.487497329711914 26.640767574310303 10.637593846321106 33.19708040318136

