#!/usr/bin/env python
# coding: utf-8

# In[175]:


import glob
import numpy as np 
import time
import math
import random
from scipy import linalg as LA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from termcolor import colored
from matplotlib import colors
import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Reading cleaned csv files

# ### User i is data_1$[i]$ in the following

# In[2]:


data_1 = [0] * 57
for i in range(len(data_1)):
    fnames = glob.glob('labeled csv Geolife/'+str(i)+'/*.csv')
    data_1[i] = np.array([np.loadtxt(f, delimiter=',')[1:] for f in fnames])
data_1 = np.array(data_1)


# In[3]:


fnames = glob.glob('labeled csv Geolife/**/*.csv')
len(fnames)


# ### Users are stacked together in data_2 below

# In[4]:


data_2 = []
fnames = glob.glob('labeled csv Geolife/**/*.csv')
for f in fnames:
    data_2.append(np.loadtxt(f, delimiter=',')[1:])
data_2 = np.array(data_2)


# In[5]:


A = np.array([len(data_2[i]) for i in range(len(data_2))])
min(A), max(A)


# ### Removing segments with length less than 1e-10 because of numerical precision

# In[6]:


data_3 = [0] * len(data_2)
h = 1e-10
c = 0
for i in range(len(data_2)):
    p1 = data_2[i][:-1]
    p2 = data_2[i][1:]
    L = ((p2[:,:2]-p1[:,:2])*(p2[:,:2]-p1[:,:2])).sum(axis =1)
    I = np.where(L > h)[0]
    J = np.where(L < h)[0]
    if len(J) > 0:
        c += 1
    p1 = p1[I]
    p2 = p2[I]
    if len(I) == 0:
        print(i)
    gamma = np.concatenate((p1, p2[-1].reshape(1,4)), 0) 
    if len(gamma) > 0:
        data_3[i] = gamma
    data_3[i] = np.array(data_3[i])
data_3 = np.array(data_3)
c


# In[7]:


A = np.array([len(data_3[i]) for i in range(len(data_3))])
min(A), max(A), np.where(A > 40000)[0]


# # Partitioning trajectories to less than 20 minutes long

# In[8]:


# 24 * 60 * (days_date('1899/12/30 2:50:06') - days_date('1899/12/30 2:20:06')) == 20 min
Time = np.zeros(len(data_3))
for i in range(len(data_3)):
    Time[i] = 24 * 60 * (data_3[i][-1][2] - data_3[i][0][2]) # = 20 minutes 
min(Time), max(Time), np.where(Time>2000)


# In[9]:


J = np.where(Time>20)[0]
len(J)


# In[10]:


def partition(trajectory):
    trajectories = []
    a = 24 * 60 * (trajectory[-1][2] - trajectory[0][2])
    if a <= 20:
        return np.array(trajectory.reshape(1, len(trajectory), 4))
    else: 
        i = 0
        while a > 20:
            j = i + 0
            val = 0
            while val < 20: 
                if i < len(trajectory) - 1:
                    temp = val + 0
                    val += 24 * 60 * (trajectory[:,2][1:][i] - trajectory[:,2][:-1][i])
                    i += 1
                else: 
                    break
            if len(trajectory[j:i-1]) > 0:
                trajectories.append(trajectory[j:i-1])
            a = a - val
        if len(trajectory[i:]) > 0:
            trajectories.append(trajectory[i:])
    trajectories = np.array(trajectories)
    return trajectories


# In[11]:


# Check to see if partitioning into less than 20 minutes worked correctly
for j in J:
    A = partition(data_3[j])
    B = np.array([24 * 60 * sum(A[i][:,2][1:] - A[i][:,2][:-1]) for i in range(len(A))])
    I = np.where(B > 20)[0]
    if len(I) > 0: 
        print(j)


# ### data_4 below is the array of trajectories having less than 20 minutes long

# In[12]:


data_4 = []
for i in range(len(data_3)):
    A = partition(data_3[i])
    for j in range(len(A)):
        data_4.append(A[j])
data_4 = np.array(data_4)


# In[13]:


data_4.shape, data_4[0].shape


# In[14]:


I = np.where(np.array([len(data_4[i]) for i in range(len(data_4))]) != 1)[0]
data_4 = data_4[I]
len(data_4)


# In[15]:


int1 = np.vectorize(int)
data_5 = []
c = 0
for i in range(len(data_4)):
    if len(set(int1(data_4[i][:,3]))) < 2: 
        data_5.append(data_4[i])
        c += 1
data_5 = np.array(data_5)
c


# In[16]:


data_6 = []
d = 0
for i in range(len(data_4)):
    if len(set(int1(data_4[i][:,3]))) == 2: 
        data_6.append(data_4[i])
        d += 1
data_6 = np.array(data_6)
d


# In[17]:


# a:b
# a is the number of labels in a trajectory
# b is the number of trajectries with a labels
D = {1:10121, 2:1671, 3:39, 4:2, 5:0}


# In[18]:


Modes = ['walk', 'bike', 'bus', 'driving', 'train']


# In[19]:


C = []
for j in range(5):
    c = 0
    for i in range(len(data_5)):
        if data_5[i][0][-1] == j:
            c += 1
    C.append(c)
print("number of trajectories of length 1 with label 0, 1, 2, 3, 4:", C)


# # Creating trajectories with 3, 4, 5, 6 labels

# ### Preparing length 2 sentences

# In[20]:


data_7 = [0] * len(data_6)
for i in range(len(data_6)):
    I = list(set(data_6[i][:,3]))
    data_7[i] = []
    J1 = np.where(data_6[i][:,3] == I[0])
    J2 = np.where(data_6[i][:,3] == I[1])
    D1 = data_6[i][J1]
    D2 = data_6[i][J2]
    data_7[i].append(D1)
    data_7[i].append(D2)
data_7 = np.array(data_7)
data_7.shape


# ### Creating sentences of length 3, 4, 5, 6 from 10039 length 1 trajectories

# In[21]:


data = []

n_1 = 1000 # number of length 1 sentences
n_2 = 1000 # len(data_7)# 1671: number of length 2 sentences
n_3 = 800 # number of length 3 sentences
n_4 = 700 # number of length 4 sentences
n_5 = 1900 # number of length 5 sentences
#n_6 = 1000 # number of length 6 sentences

for i in range(n_1):
    I = np.random.randint(0, 10039, size=1)
    data.append(data_5[I])
    
for i in range(n_2): 
    data.append(data_7[i])

for i in range(n_3):
    I = np.random.randint(0, 10039, size=3)
    data.append(data_5[I])

for i in range(n_4):
    I = np.random.randint(0, 10039, size=4)
    data.append(data_5[I])
    
for i in range(n_5):
    I = np.random.randint(0, 10039, size=5)
    data.append(data_5[I])
    
#for i in range(n_6):
#    I = np.random.randint(0, 10039, size=6)
#    data.append(data_5[I])
    
data = np.array(data)
data.shape


# ### Length of sentences

# In[22]:


print(data[0].shape, data[n_1].shape, data[n_1+n_2].shape, data[n_1+n_2+n_3].shape, 
      data[n_1+n_2+n_3+n_4].shape)


# # Functions needed for CMM

# ### Feature Mappings

# $x = (x_1, x_2, \ldots, x_n)$
# 
# $y = (y_1, y_2, \ldots, y_n) \in \{0,1,2,3,4\}^n$
# 
# If $x_i = [(a_0, b_0, t_0), \ldots, (a_m, b_m, t_m)]$, where $a_j$ is latitude, $b_j$ is longitude and $t_j$ is time, then 
# $$\displaystyle \text{length}_i = \frac{1}{m} \sum_{j=1}^m \|(a_j, b_j) - (a_{j-1}, b_{j-1})\|_2,$$
# $$\text{velocity}_i = \frac{1}{m} \sum_{j=1}^m \frac{\|(a_j, b_j) - (a_{j-1}, b_{j-1})\|_2}{t_j},$$
# $$\text{acceleration}_i = \frac{1}{m} \sum_{j=1}^m \frac{\|(a_j, b_j) - (a_{j-1}, b_{j-1})\|_2}{t_j^2}.$$
# 
# #Notice: I have divded the acceleration by 1e10 for all data. 
# 
# ### $\phi_1(x, y) = (\text{length}_i, \text{velocity}_i, \text{acceleration}_i, y_i)_{i=1}^n \in \mathbb{R}^{4n}$
# 
# ### $\phi_2(x, y) = (\text{start point}_i, \text{end point}_i, \text{length}_i, \text{velocity}_i, \text{acceleration}_i, y_i)_{i=1}^n \in \mathbb{R}^{8n}$

# ### Feature Mapping $\phi_1$

# In[23]:


def featureMapping1(data):
    Data = [0] * len(data)
    for i in range(len(data)):
        Data[i] = []
        for j in range(len(data[i])):
            D = data[i][j]
            segments = D[:,:2][1:] - D[:,:2][:-1]
            segments_lengths = np.sqrt(np.sum((segments)**2, 1))
            I = np.where(segments_lengths > 1e-8)
            D = D[I]
            if len(D) > 1:
                segments = D[:,:2][1:] - D[:,:2][:-1]
                segments_lengths = np.sqrt(np.sum((segments)**2, 1))
                length = np.sum(segments_lengths)/len(D) + 1e-10
                time = D[:,2][1:] - D[:,2][:-1] + 1e-10
                velocities = segments_lengths/time
                velocity = np.mean(velocities)
                accelerations = segments_lengths/time**2
                acceleration = np.mean(accelerations)/1e10
                y = D[0][-1]
                array = np.array([length, velocity, acceleration, y])
                Data[i].append(array)
        Data[i] = np.array(Data[i])

    Data = np.array(Data)
    K = np.array([len(Data[i]) for i in range(len(Data))])
    J = np.where(K > 0)
    Data = Data[J]
    return Data


# ### Feature Mapping $\phi_2$

# In[24]:


def featureMapping2(data):
    Data = [0] * len(data)
    for i in range(len(data)):
        Data[i] = []
        for j in range(len(data[i])):
            D = data[i][j]
            segments = D[:,:2][1:] - D[:,:2][:-1]
            segments_lengths = np.sqrt(np.sum((segments)**2, 1))
            I = np.where(segments_lengths > 1e-8)
            D = D[I]
            if len(D) > 1:
                segments = D[:,:2][1:] - D[:,:2][:-1]
                segments_lengths = np.sqrt(np.sum((segments)**2, 1))
                length = np.sum(segments_lengths)/len(D) + 1e-10
                time = D[:,2][1:] - D[:,2][:-1] + 1e-10
                velocities = segments_lengths/time
                velocity = np.mean(velocities)
                accelerations = segments_lengths/time**2
                acceleration = np.mean(accelerations)/1e10
                y = D[0][-1]
                array = np.array([length, velocity, acceleration, 
                                  D[0][0], D[0][1], D[-1][0], D[-1][1], y])
                Data[i].append(array)
        Data[i] = np.array(Data[i])

    Data = np.array(Data)
    K = np.array([len(Data[i]) for i in range(len(Data))])
    J = np.where(K > 0)
    Data = Data[J]
    return(Data)


# # Feature mapping using landmarks
# 
# $x = (x_1, x_2, \ldots, x_n)$
# 
# $y = (y_1, y_2, \ldots, y_n) \in \{0,1,2,3,4\}^n$
# 
# If $x_i = [(a_0, b_0, t_0), \ldots, (a_m, b_m, t_m)]$, where $a_j$ is latitude, $b_j$ is longitude and $t_j$ is time, then 
# $$\displaystyle \text{length}_i = \frac{1}{m} \sum_{j=1}^m \|(a_j, b_j) - (a_{j-1}, b_{j-1})\|_2,$$
# $$\text{velocity}_i = \frac{1}{m} \sum_{j=1}^m \frac{\|(a_j, b_j) - (a_{j-1}, b_{j-1})\|_2}{t_j},$$
# $$\text{acceleration}_i = \frac{1}{m} \sum_{j=1}^m \frac{\|(a_j, b_j) - (a_{j-1}, b_{j-1})\|_2}{t_j^2}.$$
# 
# #Notice: I have divded the acceleration by 1e10 for all data. 
# 
# #### Now we would like to use a feature mapping introduced in the following paper:
# 
# ``Jeff M. Phillips and Pingfan Tang. Simple distances for trajectories via landmarks. In ACM GIS SIGSPATIAL, 2019.''
# 
# Following the paper, let $q \in \mathbb{R}^2$ be a landmark and $\gamma$ be a trajectory in $\mathbb{R}^2$. We define 
# $$v_q(\gamma) = {\rm dist}(\gamma, q) = min_{p \in \gamma} \|q - p\|_2.$$
# 
# We randomly choose $m$ (here $m=20$ will be used) landmaks in $\mathbb{R}^2$ around trajectories and call them $Q$, so $Q=\{q_1, q_2, \ldots, q_m\}$. Then we define the feature mapping $v_Q$ by 
# $$v_Q(\gamma) = (v_{q_1}(\gamma), v_{q_2}(\gamma), \ldots, v_{q_m}(\gamma)) \in \mathbb{R}^m.$$
# 
# Then we combine this feature mapping with $\phi_1$ and $\phi_2$ to get the following feature mappings:
# 
# ### $\phi_3(x, y) = (v_Q(x_i), y_i)_{i=1}^n \in \mathbb{R}^{(m+1)n}$
# 
# ### $\phi_4(x, y) = (v_Q(x_i), \text{length}_i, \text{velocity}_i, \text{acceleration}_i, y_i)_{i=1}^n \in \mathbb{R}^{(m+4)n}$
# 
# ### $\phi_5(x, y) = (v_Q(x_i), \text{start point}_i, \text{end point}_i, \text{length}_i, \text{velocity}_i, \text{acceleration}_i, y_i)_{i=1}^n \in \mathbb{R}^{(m+8)n}$

# ### Landmark Feature Mapping $v_Q$

# In[25]:


def featureMap_v_Q(Q, gamma):
    
    p2 = gamma[1:]
    p1 = gamma[:-1]
    L = np.sqrt(((p2-p1)*(p2-p1)).sum(axis =1))
    II = np.where(L>10e-8)[0]
    L = L[II]
    p1 = p1[II]
    p2 = p2[II]
    w = (p1-p2)*(-1,1)/(L*np.ones((2,1))).T
    w[:,[0, 1]] = w[:,[1, 0]]
    
    dist_dot = np.sum(w * (Q.reshape(len(Q),1,2) - p1), axis=2)
    
    x = abs(dist_dot.copy())
    R = (L**2).reshape(-1,1)
    u = p1 + ((((np.sum(((Q.reshape(len(Q),1,2) - p1) * (p2 - p1)),axis=2).reshape(len(Q)
                ,-1,1,1) * (p2-p1).reshape(len(p2-p1),1,2))).reshape(len(Q),len(p1),2))/R)
    
    G = np.sqrt(np.sum((u-p1)*(u-p1), axis=2))
    H = np.sqrt(np.sum((u-p2)*(u-p2), axis=2))
    d1 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p1)*(Q.reshape(len(Q),1,2)-p1), axis=2))
    d2 = np.sqrt(np.sum((Q.reshape(len(Q),1,2)-p2)*(Q.reshape(len(Q),1,2)-p2), axis=2))

    dist = np.where(abs(G + H - L) < np.ones(len(L)) * (10e-8), x, np.minimum(d1, d2))

    j = np.argmin(dist, axis =1)
    dist_weighted = dist[np.arange(len(dist)),j]
    
    return dist_weighted.reshape(len(Q))


# ### FeatureMapping3

# In[26]:


def featureMapping3(data, Q):
    Data = [0] * len(data)
    for i in range(len(data)):
        Data[i] = []
        for j in range(len(data[i])):
            D = data[i][j]
            segments = D[:,:2][1:] - D[:,:2][:-1]
            segments_lengths = np.sqrt(np.sum((segments)**2, 1))
            I = np.where(segments_lengths > 1e-8)
            D = D[I]
            if len(D) > 1:
                segments = D[:,:2][1:] - D[:,:2][:-1]
                segments_lengths = np.sqrt(np.sum((segments)**2, 1))
                length = np.sum(segments_lengths)/len(D) + 1e-10
                time = D[:,2][1:] - D[:,2][:-1] + 1e-10
                velocities = segments_lengths/time
                velocity = np.mean(velocities)
                accelerations = segments_lengths/time**2
                acceleration = np.mean(accelerations)/1e10
                y = D[0][-1]
                array_vel_acc = np.array([y])
                mapped_v_Q_D = featureMap_v_Q(Q, D[:,:2])
                array = np.concatenate((mapped_v_Q_D, array_vel_acc), 0)
                Data[i].append(array)
        Data[i] = np.array(Data[i])

    Data = np.array(Data)
    K = np.array([len(Data[i]) for i in range(len(Data))])
    J = np.where(K > 0)
    Data = Data[J]
    return Data


# ### Feature mapping composed of $v_Q$ and featureMapping1

# In[27]:


def featureMapping4(data, Q):
    Data = [0] * len(data)
    for i in range(len(data)):
        Data[i] = []
        for j in range(len(data[i])):
            D = data[i][j]
            segments = D[:,:2][1:] - D[:,:2][:-1]
            segments_lengths = np.sqrt(np.sum((segments)**2, 1))
            I = np.where(segments_lengths > 1e-8)
            D = D[I]
            if len(D) > 1:
                segments = D[:,:2][1:] - D[:,:2][:-1]
                segments_lengths = np.sqrt(np.sum((segments)**2, 1))
                length = np.sum(segments_lengths)/len(D) + 1e-10
                time = D[:,2][1:] - D[:,2][:-1] + 1e-10
                velocities = segments_lengths/time
                velocity = np.mean(velocities)
                accelerations = segments_lengths/time**2
                acceleration = np.mean(accelerations)/1e10
                y = D[0][-1]
                array_vel_acc = np.array([length, velocity, acceleration, y])
                mapped_v_Q_D = featureMap_v_Q(Q, D[:,:2])
                array = np.concatenate((mapped_v_Q_D, array_vel_acc), 0)
                Data[i].append(array)
        Data[i] = np.array(Data[i])

    Data = np.array(Data)
    K = np.array([len(Data[i]) for i in range(len(Data))])
    J = np.where(K > 0)
    Data = Data[J]
    return Data


# ### Feature mapping composed of $v_Q$ and featureMapping2

# In[28]:


def featureMapping5(data, Q):
    Data = [0] * len(data)
    for i in range(len(data)):
        Data[i] = []
        for j in range(len(data[i])):
            D = data[i][j]
            segments = D[:,:2][1:] - D[:,:2][:-1]
            segments_lengths = np.sqrt(np.sum((segments)**2, 1))
            I = np.where(segments_lengths > 1e-8)
            D = D[I]
            if len(D) > 1:
                segments = D[:,:2][1:] - D[:,:2][:-1]
                segments_lengths = np.sqrt(np.sum((segments)**2, 1))
                length = np.sum(segments_lengths)/len(D) + 1e-10
                time = D[:,2][1:] - D[:,2][:-1] + 1e-10
                velocities = segments_lengths/time
                velocity = np.mean(velocities)
                accelerations = segments_lengths/time**2
                acceleration = np.mean(accelerations)/1e10
                y = D[0][-1]
                array_vel_acc = np.array([length, velocity, acceleration, 
                                  D[0][0], D[0][1], D[-1][0], D[-1][1], y])
                mapped_v_Q_D = featureMap_v_Q(Q, D[:,:2])
                array = np.concatenate((mapped_v_Q_D, array_vel_acc), 0)
                Data[i].append(array)
        Data[i] = np.array(Data[i])

    Data = np.array(Data)
    K = np.array([len(Data[i]) for i in range(len(Data))])
    J = np.where(K > 0)
    Data = Data[J]
    return(Data)


# ## Train-Test split function

# In[29]:


def trainTestSplit(data):
    I = []
    random.shuffle(data)
    for j in range(1,6):
        I.append(np.where([len(data[i])==j for i in range(len(Data))])[0])

    train = np.concatenate((data[I[0]][len(I[0])//3:], data[I[1]][len(I[1])//3:], 
                            data[I[2]][len(I[2])//3:], data[I[3]][len(I[3])//3:], 
                            data[I[4]][len(I[4])//3:]), 0)

    test = np.concatenate((data[I[0]][:len(I[0])//3], data[I[1]][:len(I[1])//3],
                           data[I[2]][:len(I[2])//3], data[I[3]][:len(I[3])//3],
                           data[I[4]][:len(I[4])//3]), 0)
    return train, test


# # Choosing Landmarks

# In[30]:


a, c = np.min([np.min([np.min(data[i][j][:,:2], axis=0) for j in range(len(data[i]))], 
                      axis=0) for i in range(len(data))], axis=0)
  
b, d = np.max([np.max([np.max(data[i][j][:,:2], axis=0) for j in range(len(data[i]))], 
                      axis=0) for i in range(len(data))], axis=0)

Mean = np.mean([np.mean([np.mean(data[i][k][:,:2], axis=0) for k in range(len(data[i]))], 
                      axis=0) for i in range(len(data))], axis=0)

Std = np.std([np.std([np.std(data[i][l][:,:2], axis=0) for l in range(len(data[i]))], 
                      axis=0) for i in range(len(data))], axis=0)

m = 20
Q = np.ones((m,2))

Q[:,0] = np.random.normal(Mean[0], 100*Std[0], m)
Q[:,1] = np.random.normal(Mean[1], 20*Std[1], m) 


# # LSTM Model

# In[41]:


# 1-layer LSTM with 1 FC layer
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes):

        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        
        # shape: (n_layers, batch_size, hidden_dim)
        A = (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))
       
        return A

    def forward(self, traj_span):
        
        lstm_out, self.hidden = self.lstm(traj_span.view(len(traj_span), 1, -1), 
                                          self.hidden)
        outputs = self.fc(lstm_out.view(len(traj_span), -1))
        scores = F.log_softmax(outputs, dim=1)
        
        return scores


# ## Training the model and testing
# 
# The following function first choses a model, then trains on train data and finally 
# evaluates the trained model on test data and outputs the accuracy on test data.

# In[163]:


def train_test(train_data, test_data, hidden_dim=20, num_classes=5, 
               learning_rate=0.01, n_epochs=100, d=10):
    
    start_time = time.time()
    input_dim = len(train_data[0][0][0])
    
    model = LSTM(input_dim, hidden_dim, num_classes)

    loss_function = nn.NLLLoss() #nn.CrossEntropyLoss()
    print("loss_function=", loss_function)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    print("model=", model)

    for epoch in range(n_epochs):

        epoch_loss = 0.0

        for sentence, tags in train_data:

            # zero the gradients
            model.zero_grad()

            # zero the hidden state of the LSTM, this detaches it from its history
            model.hidden = model.init_hidden()

            # forward pass to get scores
            tag_scores = model(sentence)

            # computing the loss, and gradients 
            loss = loss_function(tag_scores, tags)
            epoch_loss += loss.item()
            loss.backward()

            # updating the model parameters
            optimizer.step()

        if((epoch+1)%d == 0):
            print("Epoch: %d, loss: %1.5f" % (epoch+1, epoch_loss/len(train)))
        
    n_samples = 0
    n_correct = 0
    for trajs, labels in train_data:
        outputs = model(trajs)
        n_samples += len(labels)
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()

    acc = 100 * n_correct/n_samples
    print("n_samples=", n_samples)
    print("n_correct=", n_correct)
    print(f'train accuracy: {acc}')
    
    n_samples = 0
    n_correct = 0
    for trajs, labels in test_data:
        outputs = model(trajs)
        n_samples += len(labels)
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()

    acc = 100 * n_correct/n_samples
    print("n_samples=", n_samples)
    print("n_correct=", n_correct)
    print(f'test accuracy: {acc}')
    print(time.time() - start_time)
    
    return acc


# # Experiments

# ### Preparing data using $\phi_1$ to feed LSTM

# In[164]:


Data = featureMapping1(data)
Train, Test = trainTestSplit(Data)

train = [0] * len(Train)
for i in range(len(Train)):
    A = []
    train[i] =(torch.from_numpy(Train[i][:,:-1]).float(), 
               torch.from_numpy(Train[i][:,-1]).long())
    
test = [0] * len(Test)
for i in range(len(Test)):
    A = []
    test[i] =(torch.from_numpy(Test[i][:,:-1]).float(), 
              torch.from_numpy(Test[i][:,-1]).long())


# In[166]:


train_test(train, test, hidden_dim=20, num_classes=5, learning_rate=0.005, 
           n_epochs=100, d=10)


# ### Preparing data using $\phi_2$ to feed LSTM

# In[167]:


Data = featureMapping2(data)
Train, Test = trainTestSplit(Data)

train = [0] * len(Train)
for i in range(len(Train)):
    A = []
    train[i] =(torch.from_numpy(Train[i][:,:-1]).float(), 
               torch.from_numpy(Train[i][:,-1]).long())
    
test = [0] * len(Test)
for i in range(len(Test)):
    A = []
    test[i] =(torch.from_numpy(Test[i][:,:-1]).float(), 
              torch.from_numpy(Test[i][:,-1]).long())


# In[168]:


train_test(train, test, hidden_dim=200, num_classes=5, learning_rate=0.0008, 
           n_epochs=100, d=10)


# ### Preparing data using $\phi_3$ to feed LSTM

# In[169]:


Data = featureMapping3(data, Q)
Train, Test = trainTestSplit(Data)

train = [0] * len(Train)
for i in range(len(Train)):
    A = []
    train[i] =(torch.from_numpy(Train[i][:,:-1]).float(), 
               torch.from_numpy(Train[i][:,-1]).long())
    
test = [0] * len(Test)
for i in range(len(Test)):
    A = []
    test[i] =(torch.from_numpy(Test[i][:,:-1]).float(), 
              torch.from_numpy(Test[i][:,-1]).long())


# In[170]:


train_test(train, test, hidden_dim=100, num_classes=5, learning_rate=0.001, 
           n_epochs=100, d=10)


# ### Preparing data using $\phi_4$ to feed LSTM

# In[171]:


Data = featureMapping4(data, Q)
Train, Test = trainTestSplit(Data)

train = [0] * len(Train)
for i in range(len(Train)):
    A = []
    train[i] =(torch.from_numpy(Train[i][:,:-1]).float(), 
               torch.from_numpy(Train[i][:,-1]).long())
    
test = [0] * len(Test)
for i in range(len(Test)):
    A = []
    test[i] =(torch.from_numpy(Test[i][:,:-1]).float(), 
              torch.from_numpy(Test[i][:,-1]).long())


# In[172]:


train_test(train, test, hidden_dim=20, num_classes=5, learning_rate=0.01, 
           n_epochs=100, d=10)


# ### Preparing data using $\phi_5$ to feed LSTM

# In[173]:


Data = featureMapping5(data, Q)
Train, Test = trainTestSplit(Data)

train = [0] * len(Train)
for i in range(len(Train)):
    A = []
    train[i] =(torch.from_numpy(Train[i][:,:-1]).float(), 
               torch.from_numpy(Train[i][:,-1]).long())
    
test = [0] * len(Test)
for i in range(len(Test)):
    A = []
    test[i] =(torch.from_numpy(Test[i][:,:-1]).float(), 
              torch.from_numpy(Test[i][:,-1]).long())


# In[174]:


train_test(train, test, hidden_dim=200, num_classes=5, learning_rate=0.001, 
           n_epochs=100, d=10)


# In[ ]:




