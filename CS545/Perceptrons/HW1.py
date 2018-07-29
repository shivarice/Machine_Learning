# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 16:13:22 2017

@author: Lawrence
"""

import csv
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#generates lrate, weights and the target number from dataset
def initalize(dataset):
    lrate=np.array([0.001, 0.01,0.1])
#generates 785 weights ranging from -.5 to .5 and stores them in a dictionary
    w0=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w1=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w2=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w3=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w4=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w5=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w6=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w7=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w8=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    w9=np.array([random.uniform(-.5,.5) for x in range(len(dataset[0]))])
    weight={0:w0,1:w1,2:w2,3:w3,4:w4,5:w5,6:w6,7:w7,8:w8,9:w9}   
#takes the first item in each row and stores them these are the target value of each row
    target=np.array([float(dataset[x].pop(0)) for x in range(len(dataset))])
#takes that stored items and hot encodes them to be 1 or 0 depending on 
#perceptron target value 1=target 0=not target, t0 refers to this perceptron is only trained to number zero
#same with the other variable names 
    t0=np.array([1 if target[x]==0 else 0 for x in range(len(target))])
    t1=np.array([1 if target[x]==1 else 0 for x in range(len(target))])
    t2=np.array([1 if target[x]==2 else 0 for x in range(len(target))])
    t3=np.array([1 if target[x]==3 else 0 for x in range(len(target))])
    t4=np.array([1 if target[x]==4 else 0 for x in range(len(target))])
    t5=np.array([1 if target[x]==5 else 0 for x in range(len(target))])
    t6=np.array([1 if target[x]==6 else 0 for x in range(len(target))])
    t7=np.array([1 if target[x]==7 else 0 for x in range(len(target))])
    t8=np.array([1 if target[x]==8 else 0 for x in range(len(target))])
    t9=np.array([1 if target[x]==9 else 0 for x in range(len(target))])
#takes all target lists and stores them in a dictionary
    tar={0:t0,1:t1,2:t2,3:t3,4:t4,5:t5,6:t6,7:t7,8:t8,9:t9}
    return(lrate,weight,tar,target)
    
#takes the dataset, adds a bias at the end of each row, divide all
#values by 255 and returns it
def process(dataset):
    for x in range(len(dataset)):
        dataset[x]=dataset[x]+[255]
    dataset1=np.array(dataset, dtype=float)
    dataset1=dataset1*(1/255.0)
    return(dataset1)

    
#generates the accuracy and the predicted value from the dataset, the weight dictionary and target list generated from the 
#initalize fxn
def acc(other,weightname,target):
    target1=target.tolist()
    target1=[int(i) for i in target1]
    placeholder=[]
    predict=[]
    correct=0
    wrong=0 
    for x in range(len(other)):
        #iterates through the target list values 
        target2=target1[x]
        for y in range(len(weightname)):
            #iterate through the weight dictionary and calculate the scores in order (weights from 0 to 9 perceptron)
            score=np.dot(weightname[y],np.transpose(np.array([other[x]])))
            #stores those values
            placeholder.append(score)
        #takes the stored values and returns the position of the value that is the highest value)
        test=[i for i,x in enumerate(placeholder) if x == max(placeholder)]
        #if the return position (this position refers to which trained perceptron (ie perceptron zero says this is a zero))
        #if the return position is equal to the target value increase correct and store the value in predict
        
        if test[0]==target2:
            correct=correct+1
            predict.append(test[0])
            placeholder=[]  
        #otherwise increase wrong by 1 and append the position to predict
        else:
            wrong=wrong+1
            predict.append(test[0])
            placeholder=[]
    #calculate the accuracy at the end
    acc=correct*(float(1)/(correct+wrong))
    return(acc,predict)
#returns the score of the dot product in the form of 1 and 0 in a dictionary, must give it the dataset generated from process fxn
#and the weight dictionary generated initalize fxn
def changew(dataset,weight):
    wdict={}
    for y in range(len(weight)):
        #iterating through the weight dictionary if the dot product is greater than zero =1 otherwise zero 
        score=np.array([1 if np.dot(weight[y],np.transpose(np.array([dataset[x]])))>0 else 0 for x in range(len(dataset))])
        wdict.update({y:score})       
    return(wdict)
#creates a confusion matrix object
def cm(pred,actual):
    y_pred=pd.Series(pred, name='Predicted')
    y_actu=pd.Series(actual,name='Actual')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return(df_confusion)
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
  
    
#fxn will run everything that is needed fro each learning rate, uses fxn that have been implemented
#returns the list of accuracy of training and test set and the confusion matrix 
def runeverything(position):
    with open('mnist_train.csv','rt') as csvfile:
        original=csv.reader(csvfile, delimiter=',')
        ori=list(original)
    with open('mnist_test.csv','rt') as csvfile1:
        originaltest=csv.reader(csvfile1, delimiter=',')
        otest=list(originaltest)
    random.shuffle(ori)
    graphw=[]
    grapht=[]
    lrate,weight,tard,target=initalize(ori)
    lrate,notused,notused2,ttarget=initalize(otest)
    ori3=process(ori)
    otest2=process(otest)
    #epoch 0
    wacc,predict=acc(ori3,weight,target)
    tacc,tpredict=acc(otest2,weight,ttarget)
    wdict=changew(ori3,weight)
    graphw.append(wacc)
    grapht.append(tacc)
    #finishes the rest of the epoch
    count=0
    while count<49: 
        print count
        for b in range(len(weight)):
            #updates the weights based on algorithm
            weight[b]=weight[b]+lrate[position]*np.dot(np.array([tard[b]-wdict[b]]),ori3)
        wacc,predict=acc(ori3,weight,target)
        tacc,tpredict=acc(otest2,weight,ttarget)
        #clears the dictionary generated by changew fxn
        wdict=[]
        #generates a new dictionary using the updated weights and dataset
        wdict=changew(ori3,weight)
        graphw.append(wacc)
        grapht.append(tacc)
        count=count+1
    confuse=cm(tpredict,ttarget)
    return(graphw,grapht,confuse)

#Codes to run, Note: do not run everything at once, only as pairs as indicated by the space between codes
#otherwise the figures will lump together into one jpg. 
graphw,grapht,confuse=runeverything(0)
figure('rateoneaccshuffle.jpg','Learning Rate 0.001 Accuracy Training vs Test (shuffle)',graphw,grapht)

graphw,grapht,confuse1=runeverything(1)    
figure('ratetwoaccshuffle.jpg','Learning Rate 0.01 Accuracy Training vs Test (shufle)', graphw,grapht)

graphw,grapht,confuse2=runeverything(2)
figure('ratethreeaccshuffle.jpg','Learning Rate 0.1 Accuracy Training vs Test (shuffle)',graphw,grapht)    




#Testing codes used to create functions
#creating dictionary


#dummytraining section
#
####
     

##dummytest section
#for x in range(len(dummytest)):
#    print 'Check'
#    dummytest[x][0]=float(dummytest[x][0])
#    dummytest[x]=dummytest[x]+[255]
#    for y in range(1,len(dummytest[x])):
#        dummytest[x][y]=int(dummytest[x][y])/255.0
#scorest=[]
#for x in range(len(dummytest)):
#    dum=np.array(dummytest[x])    
#    score=int(round(np.dot(dum,weights)))
#    scorest.append(score)
#
#correct=0
#wrong=0    
#for x in range(len(scores)):
#    if scores[x]==target[x]:
#        correct=correct+1
#    else:
#        wrong=wrong+1
######
##training model
#w=w+lrate(scores-target)dummy[x][y]   
#def train(lr)


#start up code
#with open('mnist_train.csv','rt') as csvfile:
#    original=csv.reader(csvfile, delimiter=',')
#    ori=list(original)
#with open('mnist_test.csv','rt') as csvfile1:
#    originaltest=csv.reader(csvfile1, delimiter=',')
#    otest=list(originaltest)
#graphw=[]
#grapht=[]
#lrate,weight,tard,target=initalize(ori)
#print len(ori[0])
#lrate,notused,notused2,ttarget=initalize(otest)
#ori3=process(ori)
#print len(ori[0])
#otest2=process(otest)
#wacc,predict=acc(ori3,weight,target)
#tacc,tpredict=acc(otest2,weight,ttarget)
#wdict=changew(ori3,weight)
#graphw.append(wacc)
#grapht.append(tacc)
#count=49
#while count<50: 
#    print count
#    for b in range(len(weight)):        
#        weight[b]=weight[b]+lrate[0]*np.dot(np.array([tard[b]-wdict[b]]),ori3)
#    wacc,predict=acc(ori3,weight,target)
#    tacc,tpredict=acc(otest2,weight,ttarget)
#    wdict=[]
#    wdict=changew(ori3,weight)
#    graphw.append(wacc)
#    grapht.append(tacc)
#    count=count+1
  

    
    
    
#plt.plot(epoch,graphw,'r--',label='Training')
#plt.plot(epoch,grapht,'b--',label='Testing')
#plt.title('Learn Rate at 0.001 Accuracy Training vs Test')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy Percent')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
#plt.savefig('rateoneacc.jpg')
#
#        #tar=float(target[y])
#        #print(lrate[x]*(scores[y]-tar))
#weight[0]=weight[0]+lrate[0]*(wscore-target)*ori2
#
#"Use target.dot(something), in addition the columns of the first matrix must be the 
#"same amount as second matrix
# 
#for x in range(len(weight[0])):
#    weight[0][x]=weight[0][x]+lrate[0]*(np.array([wscore-target]))*ori2[]
#test3=(np.array([wscore-target])).dot(ori2)
#np.dot(weight[0],np.transpose(np.array([ori2[0]]))
#weight[0].shape
#np.transpose(np.array([ori2[0]])).shape
#
#
#score=np.around(np.dot(weight[0],np.transpose(np.array([ori2[0]]))))
#round(np.dot(weight[0],np.transpose(np.array([ori2[0]]))))
#weight[0].shape
#-1.81813570e+02
#
#"Two different parts one is updating the weights which only use 1 or 0, must hot encoded the target values either 0 or 1 based on what perceptron model is trained for
#"The second part is then comparing highest dot product which determines which perceptron number is the best prediction and then compare to the target value. then cacluate the accuracy 