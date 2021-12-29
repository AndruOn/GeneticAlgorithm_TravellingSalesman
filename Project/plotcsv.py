import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('plot.csv')

iter = df['Iteration']
bestCost = np.array(df['BestValue'])
meanCost = df['MeanValue']
meamMutation = df['MeanMutation']
meamCrossover = df['MeanCrossover']
k_selection = df['k_selection']
k_elimination = df['k_elimination']
diversityIndicator = df['DiversityIndicator']

plt.figure(figsize=(20, 8), dpi=70)

fig, axs = plt.subplots(2,2)
#MeanCost and bestCost
axs[0, 0].plot(iter, bestCost,label = "Best Path Cost")
axs[0, 0].plot(iter, meanCost,label = "Mean Cost")
axs[0, 0].set_title('costs')
axs[0, 0].set_ylabel('Cost of the Path')
index_firstRealSolution = np.where(bestCost < 10**20 * 1.0)[0][0]
#print("index_firstRealSolution:",index_firstRealSolution,"np.min(bestCost < 10^20)")
axs[0, 1].set_ylim(top= 1.0)
axs[0, 0].legend()
#Diversity 
axs[0, 1].plot(iter, diversityIndicator,label = "diversity Indicator")
axs[0, 1].plot(iter, np.full(iter.size,np.mean(diversityIndicator)), '-.', label = "diversity mean")
axs[0, 1].set_title('Diversity')
axs[0, 1].set_ylabel('DIversity')
axs[0, 1].set_ylim(bottom=0.0, top= np.max(diversityIndicator)*1.1)
axs[0, 1].legend()

"""
#Second half of iter mean best
axs[0, 1].plot(iter, bestCost,label = "Best Path Cost")
axs[0, 1].plot(iter, meanCost,label = "Mean Cost")
axs[0, 1].set_title('end of costs')
axs[0, 1].set_ylabel('Cost of the Path')
axs[0, 1].legend()
startEnd_index = iter.size - 30
axs[0, 1].set_xlim(left=startEnd_index,right = iter.size + 1)
axs[0, 1].set_ylim(top=max(meanCost[startEnd_index:]) + 2500, bottom=min(bestCost[startEnd_index:]) -2500)
"""
#k values for selection and elimination
axs[1, 0].plot(iter, k_selection, label = "k_selection")
axs[1, 0].plot(iter, k_elimination, label = "k_elimination")
axs[1, 0].set_title("k values")
axs[1, 0].set_ylabel('k value')
axs[1, 0].legend()
#Mutation and crossover probability
axs[1, 1].plot(iter, meamMutation, label = "meamMutation")
axs[1, 1].plot(iter, meamCrossover, label = "meamCrossover")
axs[1, 1].set_title('Mutation and crossover probability')
axs[1, 1].set_ylabel('Probability')

axs[1, 1].set_ylim(0.0,1.0)
axs[1, 1].legend()


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    #ax.label_outer()
    ax.set_xlabel('Iterations')

#padding between subplots
fig.tight_layout(pad=2.0)

plt.savefig("testPlot.pdf")

#----------------------------------------------------------

# Python program to create
# a pdf file
 
from fpdf import FPDF
# save FPDF() class into
# a variable pdf
pdf = FPDF()  
  
# Add a page
pdf.add_page()
  
# set style and size of font
# that you want in the pdf
pdf.set_font("Arial", size = 10)
 
# open the text file in read mode
f = open("lastParams.txt", "r")
 
# insert the texts in pdf
for x in f:
    pdf.cell(200, 5, txt = x, ln = 1, align = 'C')
  
# save the pdf with name .pdf
pdf.output("lastParams.pdf")  


from PyPDF2 import PdfFileMerger, PdfFileReader
merger = PdfFileMerger()

merger.append(PdfFileReader(open("testPlot.pdf", 'rb')))
merger.append(PdfFileReader(open("lastParams.pdf", 'rb')))

merger.write("finalPlot.pdf")

import os
os.remove("testPlot.pdf") 
os.remove("lastParams.pdf") 
os.remove("lastParams.txt") 
