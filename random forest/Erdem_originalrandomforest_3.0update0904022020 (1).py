#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 11:32:41 2020

@author: internet
"""

import random
import csv
import pandas as pd
import numpy as np
 #import seaborn as sn
 #import math
 #from numpy import array
 #from math import *
 #from sklearn.model_selection import train_test_split
 #from sklearn.ensemble import RandomForestClassifier
 #from sklearn import metrics

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())



#step0: Importing the dataset
print("\n\n#step0:importing the dataset")
df=np.array(["Erdem_data.csv"])
df=pd.read_csv('Erdem_data.csv', skiprows=[])
   #df.at[5,"3"]=30
print(df)



#step1:random forest
print("\n\n#step1:random forest")
rw=rowcolumn=-1
rwmt=rowcolumnmatrix=[]
row=[]
with open('Erdem_data.csv') as file:
    dfr = csv.reader(file)
    for row in dfr:
        print(" ",row)
        row.append(row)
        rw=rw+1
        rwmt.append(rw)
gm=generalmatrix=np.zeros((rw,rw))
rfdtn=random_forest_decision_tree_number=10#oooooooooooooooooooooooooooo
print("\n number of columns:",rwmt,"\n")
i=0
rw2=rw

while i<rfdtn:
    rnmt=randomnodematrix=[]
    rtfmt=randomtruefalsematrix=[]
    while rw>-1:
        rn=randomnodes=random.choice((rwmt))
        rtf=randomtruefalse=random.choice([1,0])
        rnmt.append(rn)
        rtfmt.append(rtf)
        rw=rw-1
        
    seen = {}
    dupes = []
    i2=0
    i3=0
    dupesmt=[]
    for x in rnmt:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
                i3=i2
                dupesmt.append(i3)
            seen[x] += 1
        i2=i2+1
    print(" duplicate numbers:",dupes,"\n seen:",seen,"\n dublicate coordinates on random forest:",dupesmt)
    i4=0
    realrnmt=[]
    while dupesmt[0]>0:
        i4=i4+1
        realrnmt.append(rnmt[i4])
        dupesmt[0]=dupesmt[0]-1
    realrnmt.pop(-1)
    print(" random forest without deleting dublicates:",rnmt,"\n random forest true false without deleting dublicates",rtfmt,)
    print(" random forest:",realrnmt,"\n random forest true false:",rtfmt)

    for i5 in realrnmt:
        print(rtfmt[i5])
        if i5==10:
            break
        dfcol=pd.read_csv('Erdem_data.csv', usecols=[i5])
        print(dfcol)
        a=np.loadtxt(open("Erdem_data.csv", "rb"), delimiter=",", skiprows=1,usecols=[i5])
        print(" matrix with spesific column:",a)
        c=0
        t=[]
        f=[]
        for x in a:
            if x==1:
                t.append(c)
                print(" true:",c)
            if x==0:
                f.append(c)
                print(" false:",c)
            c=c+1
        print(" true rows:",t)
        print(" false rows:",f)
        
        short=[]
        if rtfmt[i5]==1:
            short=f
        if rtfmt[i5]==0:
            short=t
        print(" shorts:",short,"\n")
        for i6 in short:
            for i7 in short:
                if i6==i7:
                    break
                gm[i6][i7]=1+gm[i6][i7]
        break
    print(" ------------------------")
    rw=rw2
    i=i+1



#step2:general matrix
print("\n\n#step2:general matrix")
print(symmetrize(gm))



#step3:distance matrices
print("\n\n#step3:distance matrix")
print(symmetrize(gm)/9)
   

   
#step4:heat map
print("\n\n#step4:heat map")
print("  system has not include sklearn, seaborn and pyplot(matplotlib)")



#step5:what is your input
print("\n\n#step5:what is your input")
inpt=[]#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
outpt=0
print("  input:",inpt,"[?]")
inptnum=[]
for xno in inpt:
    if xno=="yes":
        inptnum.append(1)
    else:
        inptnum.append(0)
print("  converted digits:",inptnum)
print("  answer below the code\n")

   #if outpt<0:
   #    print("  heart disease result: no")
   #else:
   #    print("  heart disease result: yes")
   #print("  yes-no number of trees in",i,":",outpt,"\n  result possibility:",abs((outpt/rfdtn)*100),"%")

#Erdem Erbaba