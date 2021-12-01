import torch
import random
from torch.autograd import Variable
from dataLoader import read_data
from dataLoader import read_label

DATA_LENGTH = 13
EPOCH = 5000
K = 10
BATCH_SIZE = 128

data_med = read_data("traindata.txt")
label_med = read_label("trainlabel.txt")

LEN = len(label_med)


def sample(list, batch_size):

    tensor1 = torch.tensor(list)
    a, b = tensor1.shape
    index = torch.LongTensor(random.sample(range(a), batch_size))
    tensor2 = torch.index_select(tensor1, 0, index)
    # print(tensor2)
    # tensor1 = tensor1[torch.randperm(tensor1.size(0))]
    # tensor, useless = tensor1.split(length, 0)

    data, label = tensor2.split(13, 1)
    label = torch.squeeze(label)
    #print(data.size(), label.size())

    return data, label

vote_nn = []

for k in range(K):
    train = []
    setlist = []
    validation_data = []
    validation_label = []

    data_med = read_data("traindata.txt")
    label_med = read_label("trainlabel.txt")
    for i in range(LEN):
        setlist.append(data_med[i])
        setlist[len(setlist) - 1].append(label_med[i])

    data_set, label_set = sample(setlist, len(setlist))
    data_med = data_set.tolist()
    label_med = label_set.tolist()

    for i in range(LEN):

        if LEN * (k / K) <= i < LEN * ((k + 1) / K):
            validation_data.append(data_med[i])
            validation_label.append(label_med[i])
        else:
            #print(i, len(data_med[i]), data_med[i])
            train.append(data_med[i])
            train[len(train) - 1].append(label_med[i])


    # print(len(data), len(label), len(validation_data), len(validation_label))
    validation_data = torch.tensor(validation_data)
    validation_label = torch.tensor(validation_label)

    validation_data, validation_label = Variable(validation_data), Variable(validation_label)

    net = torch.nn.Sequential(
        #torch.nn.Sigmoid(),
        torch.nn.Linear(DATA_LENGTH, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
        torch.nn.Softmax(dim=1)

        # torch.nn.Linear(DATA_LENGTH, 16),
        # torch.nn.ReLU(),
        # torch.nn.Linear(16, 64),
        # torch.nn.ReLU(),
        # torch.nn.Linear(64, 512),
        # torch.nn.ReLU(),
        # torch.nn.Linear(512, 64),
        # torch.nn.ReLU(),
        # torch.nn.Linear(64, 16),
        # torch.nn.ReLU(),
        # torch.nn.Linear(16, 2),
        # torch.nn.Softmax(dim=1)
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)  # 随机梯度下降
    loss_func = torch.nn.CrossEntropyLoss()


    for t in range(EPOCH):
        data, label = sample(train, BATCH_SIZE)
        data, label = Variable(data), Variable(label)

        #print(data.shape)
        out = net(data.float())
        #print(out.shape)
        loss = loss_func(out, label.long())  # must be (1. nn output, 2. target), the target label is NOT one-hotted

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    predict = net(validation_data)
    score = 0
    predict = torch.max(predict, 1).indices
    print(predict)
    for i in range(len(predict)):
        if predict[i] == validation_label[i]:
            score = score + 1

    # print(predict,validation_label)
    score = score / len(predict)
    vote_nn.append([net, predict, score])
    print("In %d validation: acc = %f" % (k + 1, score))

test = read_data("testdata.txt")
test_tensor = torch.tensor(test)
ticket = []
for i in range(len(vote_nn)):
    same = False
    vote_net = vote_nn[i][0]
    predict = vote_nn[i][1]

    for j in range(len(predict)-1):
        if predict[j] != predict[j+1]:
            break
        if j == len(predict)-1:
            same = True
    #print(same)
    if not same:
        ticket.append(torch.max(vote_net(test_tensor), 1).indices)

answer = []
for i in range(len(test)):
    vote0 = 0
    vote1 = 0
    for j in range(len(ticket)):
        if ticket[j][i] == 0:
            vote0 = vote0 + vote_nn[j][2]
        if ticket[j][i] == 1:
            vote1 = vote1 + vote_nn[j][2]
    if vote0 > vote1:
        answer.append(0)
    if vote1 > vote0:
        answer.append(1)
    if vote0 == vote1:
        if random.random() > 0.5:
            answer.append(0)
        else:
            answer.append(1)

    #print(vote0, vote1)

print(answer)

predict_score = []
record_score= []

for i in range(len(vote_nn)):
    name1 = 'nn'+str(i)+'.pkl'
    name2 = 'nn'+str(i)+'_params.pkl'

    torch.save(vote_nn[i][0], name1)
    torch.save(vote_nn[i][0].state_dict(), name2)

    predict_score.append(vote_nn[i][1].tolist())
    record_score.append(vote_nn[i][2])

f = open('record', 'w', encoding='utf-8')

f.write(str(predict_score))
f.write(str(record_score))
f.close()
# test = torch.tensor([42, 1, 3, 120, 240, 1, 0, 194, 0, 0.8, 3, 0, 7])
# print(net(data))
# torch.save(net, 'nn.pkl')
# torch.save(net.state_dict(), 'nn_params.pkl')
