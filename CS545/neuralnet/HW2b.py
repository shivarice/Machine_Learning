# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:22:00 2017

@author: Lawrence Hsu
"""


import csv
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#input weight generate
def weightgen(rows,columns):
    dictw=np.random.uniform(low=-.05,high=.05, size=(rows,columns))
    return(dictw)
#sigmoid fxn
def sigmoid (x): 
    return 1/(1 + np.exp(-x))

#function for calculating the hidden layer nodes
def hlayer(dataset,weight):    
    hidden=np.dot(np.array([dataset]),np.transpose(weight))
    x=np.ones((1,1))
    hidden=np.concatenate((hidden,x),axis=1)
    return(hidden)    
    
#function for calculating the output nodes
def outlayer(dataset,weight):
    output=np.dot(dataset,np.transpose(weight))
    return(output)

def oerror(output,tard):
    #note to self * is different from dot as the * uses the hadamard product where elemnts in the same position
    #are multipled, the conditions is the two matrices must be of the same shape
    oerror=output*(1-output)*(tard-output)
    return(oerror)

#calculate the delta in hidden weights
def deltahidden(lrate,oerror1,hidlay):
    place=lrate*np.transpose(oerror1).dot(hidlay)
    return(place)

#calculate the hidden error terms
def herror(oerror,hidlay,hweight):
    hiddenerror=np.delete(hidlay,len(hidlay[0])-1,1)*(1-np.delete(hidlay,len(hidlay[0])-1,1))*(np.dot(oerror,np.delete(hweight,len(hweight[0])-1,1)))
    return(hiddenerror)
#calculate the delta in input weights
def deltainput(lrate,hiddenerror,inputset):
    place=lrate*np.transpose(hiddenerror).dot(np.array([inputset]))
#    for x in range(len(hiddenerror)):
#        place=np.transpose(np.array([hiddenerror[x]])).dot(np.array([inputset[x]]))
#        inputweight=inputweight+lrate*place+momentum*iteration
    return(place)
    #inputweight=inputweight+lrate*np.transpose(hiddenerror).dot(inputset)+momentum*iteration


#generates lrate, weights, momentum and the target number from dataset
def initalize(dataset,rows,columns):
    lrate=np.array([0.1])
    momentum=np.array([0,0.25,0.5,0.9])
#generates 785 weights ranging from -.5 to .5 and stores them in a dictionary
    weight=weightgen(rows,columns)
#takes the first item in each row and stores them these are the target value of each row
    targe=[float(dataset[x].pop(0)) for x in range(len(dataset))]
#takes that stored items and hot encodes them to be 1 or 0 depending on 
#perceptron target value 1=target 0=not target, t0 refers to this perceptron is only trained to number zero
#same with the other variable names 
    tar=target(targe)
#takes all target lists and stores them in a dictionary
    return(lrate,weight,tar, momentum)

#takes the dataset, adds a bias at the end of each row, divide all
#values by 255 and returns it
def process(dataset):
    for x in range(len(dataset)):
        dataset[x]=dataset[x]+[255]
    dataset1=np.array(dataset, dtype=float)
    dataset1=dataset1*(1/255.0)
    return(dataset1)
#runs through the target list generated from initalize fxn and generates 
#a numpy array with 0.1 and 0.9
#each row represents the target value for each training example
def target(target):
    target1=np.array([[0,1,2,3,4,5,6,7,8,9]])
    targetlist=[]
    for x in range(len(target)):
        for y in range(10):
            if y==target[x]:
                targetlist.append(0.9)
            else:
                targetlist.append(0.1)
        targetlist2=np.array([targetlist])
        target1=np.concatenate((target1,targetlist2),axis=0)
        targetlist=[]
    target2=np.delete(target1,0,0)
    return(target2)
#calculates average
def average(x,y):
    return x*(1/(x+float(y)))
#calculates the accuracy
def accuracy(prediction,targets):
    correct=0
    wrong=0
#runs through the prediction and total the number of corrects and wrongs
    for b in range(len(prediction)):
        if prediction[b]==targets[b]:
            correct=correct+1
        else:
            wrong=wrong+1
#use average fxn to calculate overall accuracy
    score=average(correct,wrong)
    return(score)
#creates graphs, you must give it a then name of the save figh with extension, title of the graph, the lists of accuracy
#of training and test 
def figure(savename,titlename,graphw,grapht):
    epoch=list(range(50))
    plt.plot(epoch,graphw,'r--',label='Training')
    plt.plot(epoch,grapht,'b--',label='Testing')
    plt.title(titlename)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Percent')
    plt.legend(loc=0, borderaxespad=0)
    plt.savefig(savename)

#creates a confusion matrix object
def cm(pred,actual):
    y_pred=pd.Series(pred, name='Predicted')
    y_actu=pd.Series(actual,name='Actual')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return(df_confusion)



#only use these lines of codes at the beginning 
with open('mnist_train.csv','rt') as csvfile:
    original=csv.reader(csvfile, delimiter=',')
    ori=list(original)
with open('mnist_test.csv','rt') as csvfile1:
    originaltest=csv.reader(csvfile1, delimiter=',')
    otest=list(originaltest)
lrate,weight,tar,momentum=initalize(ori,20,785)
notused,tiweight,ttar,notused2=initalize(otest,20,785)
otest2=process(otest)
ori2=process(ori)

tline=np.where(tar>.5)
tline2=tline[1].tolist()
ttline=np.where(ttar>.5)
ttline2=ttline[1].tolist()


#Use these lines of code to reinitalize the weights or change how many needed to be created
#for each experiment
hweight=weightgen(10,21)
weight=weightgen(20,785)


#run block of code for each trial with adjustables
epoch=0
pred=np.array([0])
pred1=np.array([0])
prehid=0
preout=0
tgraph=[]
wgraph=[]
while epoch<50: 
    print epoch
    pred=np.array([0])
    pred1=np.array([0])
    for x in range(len(ori2)):
        hidlay=sigmoid(hlayer(ori2[x],weight))
        outlay=sigmoid(outlayer(hidlay,hweight))
        oerror1=oerror(outlay,tar[x])
        herror2=herror(oerror1,hidlay,hweight)
        dhidden=deltahidden(lrate,oerror1,hidlay)
        dinput=deltainput(lrate,herror2,ori2[x])
#momentum is an array with index from 0 to 3, change the index to switch to another 
#momentum value 
        hweight=hweight+dhidden+(momentum[3]*prehid)
        weight=weight+dinput+(momentum[3]*preout)
        prehid=np.copy(dhidden)
        preout=np.copy(dinput)
    epoch=epoch+1
    for y in range(len(otest2)):
        thidlay=sigmoid(hlayer(otest2[y],weight))
        toutlay=sigmoid(outlayer(thidlay,hweight))
        pred=np.append(pred,np.argmax(toutlay))
    test_pred1=np.delete(pred,0,0).tolist()
    tgraph.append(accuracy(test_pred1,ttline2))
    for z in range(len(ori2)):    
        hidlay2=sigmoid(hlayer(ori2[z],weight))
        outlay2=sigmoid(outlayer(hidlay2,hweight))
        pred1=np.append(pred1,np.argmax(outlay2))
    pred2=np.delete(pred1,0,0).tolist()
    wgraph.append(accuracy(pred2,tline2))    




    
    
