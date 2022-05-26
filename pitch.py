#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:58:16 2022

@author: kgraham
"""

# Python packages to import (via Anaconda environment)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import permutation_test_score as pv
from sklearn.metrics import classification_report as cr
from sklearn import tree

# open csv files
df = pd.read_csv('/Users/kgraham/pitchclassificationtrain.csv')

dftrain = df[ ['pitchid', 'pitcherid', 'yearid', 'height', 'ballSpeed', 'curve_X',
       'curve_Z', 'releasePoint_X', 'releasePoint_Y', 'releasePoint_Z',
       'ballSpin'] ]
dftype = df['type']

# train and test dataframes
x1, x2, y1, y2 = train_test_split( dftrain, dftype )    # default 75 % train, 25 % test

# input dataset for models, removed pitchid, pitcherid, yearid, height, and rpY to not introduce unnecessary variables
x1 = x1[['ballSpeed', 'curve_X', 'curve_Z', 'releasePoint_X', 'releasePoint_Z', 'ballSpin']]
x2 = x2[['ballSpeed', 'curve_X', 'curve_Z', 'releasePoint_X', 'releasePoint_Z', 'ballSpin']]


#----------------Model Choice 1: KNN-----------------#
# number of "neighbors" (categories), find number of pitch types
k = len( y1.unique() )

# pipeline: K-nearest Neighbors Classification method with variable standardization
pipe = make_pipeline( StandardScaler(), kn(n_neighbors=k) )
model = pipe.fit(x1, y1)

# model prediction of pitch type
prediction_knn = model.predict(x2)
prediction_knn = pd.Series(prediction_knn)

# cross validation: prevents model overfitting
val = cross_val_score(model, x2, y2, cv=5)
pval = pv(model, x2, y2, cv=5)[2]
mean = np.round( val.mean(), decimals=2 )
stdev = np.round( val.std(), decimals=2 )
pval = np.round( pval, decimals=3 )
print( 'KNN cross-val score of {}, stdev {}, p-value {}'.format(mean, stdev, pval) )
print( cr(y2, prediction_knn) )  # includes precision and recall for each pitch type

#----------------Model Choice 2: Decision Tree-----------------#
# no standardization needed for tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(x1, y1)

# for csv output
predicted_dt = dt.predict(x2)
predicted_dt = pd.Series(predicted_dt)

# model cross validation
val2 = cross_val_score(dt, x2, y2, cv=5)
pval2 = pv(dt, x2, y2, cv=5)[2]
mean2 = np.round( val2.mean(), decimals=2 )
stdev2 = np.round( val2.std(), decimals=2 )
pval2 = np.round( pval2, decimals=3 )
print( 'Decision Tree cross-val score of {}, stdev {}, p-value {}'.format(mean2, stdev2, pval2) )
print( cr(y2, predicted_dt) )


#----------------model predictions to csv-------------------#

# reset index of y2 for comparison to model predictions
y_out = y2.reset_index()

# combine to dataframe for easy validation
out = pd.concat( [y_out, prediction_knn, predicted_dt], axis=1 )
out.columns = ['pitch_index', 'actual', 'predicted_KNN', 'predicted_DT']
#out.to_csv('/Users/kgraham/pitchtype_prediction.csv')


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Figures for data exploration

# Pitch Type versus Pitcher: who throws what pitch types?
def ptp():
    plt.figure()
    plt.plot(df.type, df.pitcherid, 'X', markersize=15, color='navy')
    plt.xticks([1,2,3,4,5,6,7,8,9,10], fontsize=14)
    plt.yticks([1,2,3,4,5], fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid('--',alpha=0.5)
    plt.xlabel('Pitch Type', fontsize=14)
    plt.ylabel('Pitcher ID', fontsize=14)
    plt.title('Who throws what pitch type?', fontsize=14)
    
def boxplot(var):  # boxplots for pitch type versus variable
    # input var as 'variable' from df
    data = df[[var, 'type']]
    types = [2,3,4,7,8,9,10]
    medianprops = dict(linewidth=2.5, color='navy')
    plt.figure()
    plt.subplot(111)
    total = []
    for x, i in enumerate(types):
        total.append( data[var][data.type==i] )
    plt.boxplot(total, showfliers=False, medianprops=medianprops, patch_artist=True, boxprops=dict(facecolor='lightgray' ))
    plt.xticks( range(1,7+1), types, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid('--',alpha=0.5)
    plt.xlabel('Pitch Type', fontsize=14)
    plt.ylabel('{}'.format(var), fontsize=14)
    plt.title('Box and Whisker Plot', fontsize=14)

def scatter(var1, var2):    # scatter plots for each pitch type
    data = df[[var1, var2, 'type']]
    types = [2,3,4,7,8,9,10]
    color = iter(cm.tab10(np.linspace(0, 1, len(types))))
    plt.figure()
    for t in types:
        c = next(color)
        plt.plot( data[var1][data.type==t], data[var2][data.type==t], '.', label=t, color=c, alpha=0.5 )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid('--',alpha=0.5)
    plt.xlabel('{}'.format(var1), fontsize=14)
    plt.ylabel('{}'.format(var2), fontsize=14)
    plt.title('{} vs. {}'.format(var1, var2), fontsize=14)
    plt.legend(markerscale=3, title='Pitch Type')
    #plt.savefig('/Users/kgraham/figs/{}vs{}.png'.format(var1, var2))

