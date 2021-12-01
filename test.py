import torch
from dataLoader import read_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.autograd import Variable

# make fake data
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

nn_vote = []
answer = []
ticket = []
test = read_data("testdata.txt")
test_tensor = torch.tensor(test)

for i in range(10):
    name = 'nn'+str(i)+".pkl"
    nn_vote.append(torch.load(name)) # 加载

# for i in range(len(nn_vote)):
#     same = False
#     vote_net = nn_vote[i]
#     predict = nn_vote[i]
#
#     for j in range(len(predict)-1):
#         if predict[j] != predict[j+1]:
#             break
#         if j == len(predict)-1:
#             same = True
#     #print(same)
#     if not same:
#         ticket.append(torch.max(vote_net(test_tensor), 1).indices)

# for i in range(len(test)):
#     vote0 = 0
#     vote1 = 0
#     for j in range(len(ticket)):
#         if ticket[j][i] == 0:
#             vote0 = vote0 + vote_nn[j][2]
#         if ticket[j][i] == 1:
#             vote1 = vote1 + vote_nn[j][2]
#     if vote0 > vote1:
#         answer.append(0)
#     if vote1 > vote0:
#         answer.append(1)
#     if vote0 == vote1:
#         if random.random() > 0.5:
#             answer.append(0)
#         else:
#             answer.append(1)