# http://pytorch.org/
# https://keras.io/
!pip install -q keras
from os import path
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import optimizers
from sklearn.cross_validation import train_test_split
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.3.0.post4-{platform}-linux_x86_64.whl torchvision
!pip install tqdm
import torch

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
from tqdm import tqdm, trange
def V(x):   # converts a numpy array to a pyTorch variable
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=True)
def C(x):   # same as above; but this is for y0 (the actual value), not y (the NN-output value); pyTorch actually forces me to do this...
    return Variable(torch.FloatTensor(np.array(x)), requires_grad=False)
def N(x):   # converts back
    return x.data.cpu().numpy()

Nsamples = 20000
max_Nfeatures = 100
epochs = 300

# generate data
Xs = []
ys = []
for isample in trange(Nsamples):
    # a random number of features
    Nfeatures = np.random.randint(max_Nfeatures-40) + 40

    # generate X
    # intermediateX example: shape [50, 3], content [[-25,25,44], [-10,30,40], ....]
    intermediateX = np.random.rand(Nfeatures, 3) * 50 - 50 * np.random.rand(3)
    # X example: shape [2500, 6], content [[-25,25,44,-25,25,44], [-25,25,44,-10,30,40], ...]
    X = np.zeros((Nfeatures, Nfeatures, 6))
    X[:,:,:3] = np.expand_dims(intermediateX, 1)
    X[:,:,3:] = np.expand_dims(intermediateX, 0)
    X = X.reshape(-1,6)
    X = X * np.mgrid[1:2:len(X)]   # a modification that would seem meaningless from your point of view... just keep it...
    # Note: all these is similar to below; but I found my data-generating method to generate much-harder-to-train data
    # X = np.random.rand(Nfeatures * Nfeatures, 6)

    # generate y
    origin = V([0,0,0])
    # firstly, z=f(x)
    X1 = V(X[:,:3]) - origin   # example shape: (2500, 3); origin=0 so ignore origin for now
    X2 = V(X[:,3:]) - origin
    R1 = torch.norm(X1, dim=1)       # example shape: (2500, 1)
    R2 = torch.norm(X2, dim=1)
    z = 1 / R1 / R2      # example shape: (2500, 1)
    z = torch.sum(z) * 1e4        # example shape: 1
    # then y=g(z). why not just ask you to train z? because training y is much more difficult than z.
    y = N(torch.autograd.grad(z, origin, create_graph=True)[0])

    Xs.append(np.mean(X, 0))
    ys.append(y / (len(X)))


print("\n")
print(Xs[0].shape)
print(y.shape)
print(y)
print(X[0])
print(Xs[0])
print(ys[0])


X_train, X_test, y_train, y_test = train_test_split(Xs, ys, train_size = 0.8)
print(type(X_train))

print("train")
print(len(X_train))
print(X_train[0].shape)
model = Sequential()
model.add(Dense(units=400, input_dim=6))
model.add(Activation('relu'))
model.add(Dense(units=200))
model.add(Activation('relu'))
model.add(Dense(units=100))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Activation('relu'))
model.add(Dense(units=3))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer='sgd',metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train), epochs=100, batch_size=50, verbose=2)

loss_and_metrics = model.evaluate(np.array(X_test), np.array(y_test), batch_size=100)

classes = model.predict(np.array(X_test), batch_size=1)

print(classes[:10])
print(y_test[:10])
"""
# neural net
net = nn.Sequential(
    nn.Linear(6, 192),
    nn.ReLU(),
    nn.Linear(192, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
optimizer = optim.Adam(net.parameters(), lr=1E-4)
criterion = nn.MSELoss()
net.train()

# train
for epoch in trange(epochs * Nsamples): # using trange instead of range produces a progressbar
    # randomly select a sample
    isample = np.random.randint(0, Nsamples)
    y0 = C(ys[isample])

    # calculate neural net output
    X = V(Xs[isample]) - torch.cat([origin, origin])
    z = torch.sum(net(X))
    # z = torch.sum(1 / torch.norm(X[:,:3], dim=1) / torch.norm(X[:,3:], dim=1) * 1e4)  # this is 100% accurate
    y = torch.autograd.grad(z, origin, create_graph=True)[0]

    # and perform train step
    loss = criterion(y,y0)
    optimizer.zero_grad()   # suggested trick
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        tqdm.write('%s\t%s\t%s\t%s' %(epoch, N(loss)[0], N(y0)[0], N(y)[0]))
"""

