import pandas as pd
import matplotlib.pyplot as plt
import csv


df = pd.read_csv('plot.csv')

iter = df['Iteration']
bestCost = df['BestValue']
meanCost = df['MeanValue']

fig, ax = plt.subplots()
ax.plot(iter, bestCost,label = "Best Path Cost")
ax.plot(iter, meanCost,label = "Mean Cost")
ax.set_xlabel('Iterations')  # Add an x-label to the axes.
ax.set_ylabel('Cost of the Path (value of the fitness function)')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.set_title('Convergence for tour29.csv')
ax.legend()
plt.show()