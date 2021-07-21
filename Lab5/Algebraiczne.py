import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.metrics import davies_bouldin_score
import csv
from math import pi


with open("ZTS_data.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    next(csv_reader)
    X = []
    y = []
    for row in csv_reader:
        X.append(row[1:4])
        y.append(row[4])

for i in range(len(X)):
    for j in range(len(X[0])):
        X[i][j] = float(X[i][j])

for i in range(len(y)):
    y[i] = float(y[i])

X_std = np.array(X)
X_std = X_std.mean(axis=0) - X_std


days = 5
backward = 100

X_train = np.array(X_std[0:(len(X_std) - days)])
y_train = np.array(y[0:(len(X_std) - days)])
X_test  = np.array(X_std[(len(X_std) - days):len((X_std))])
y_test  = np.array(y[(len(X_std) - days):len((X_std))])

# X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.1)

def wagi(X, p, alfa_0, neuron_method, alfa_method, epoch):
    N = X.shape[0]
    l = X.shape[1]
    w = np.ones([p, l]) / np.sqrt(p)
    Xp = np.zeros(N) - 1
    T = N * epoch
    C = -np.log(0.001 / alfa_0) / T
    C2 = 1
    C1 = 0.001 * (C2 + T)

    for k in range(epoch):
        for j in range(N):
            m = -1
            if neuron_method == 0:
                max = float('-inf')
                for i in range(p):
                    x = np.sum(w[i] * X[j])
                    if x > max:
                        max = x
                        m = i
            if neuron_method == 1:
                min = float('inf')
                for i in range(p):
                    x = np.sqrt(np.sum((w[i]-X[j]) ** 2))
                    if x < min:
                        min = x
                        m = i
            if neuron_method == 2:
                min = float('inf')
                for i in range(p):
                    x = np.sqrt(np.sum(np.fabs(w[i] - X[j])))
                    if x < min:
                        min = x
                        m = i

            if alfa_method == 0:
                alfa = alfa_0 * (T - k * N + j + 1) / (T + 1)
            if alfa_method == 1:
                alfa = alfa_0 * np.exp(-C * k)
            if alfa_method == 2:
                alfa = C1 / (C2 + k)

            w[m] += alfa * (X[j] - w[m])
            w[m] /= np.sqrt(np.sum(w[m] ** 2))
            Xp[j] = m

    return w, Xp

def fi(x, r):
    return np.exp(-(x/r)**2)


r = max(np.linalg.norm(X_train[i, :] - X_train[j, :]) for i in range(X_train.shape[0]) for j in range(i)) #liczenie promienia
r = r/2
p = 16 #liczba klastrów
c, _ = wagi(X_train, p, 1, 2, 2, 100)

FI = np.zeros([X_train.shape[0], p])

for i in range(X_train.shape[0]):
    for j in range(p):
        FI[i][j] = fi(np.sqrt(np.sum((X_train[i] - c[j])**2)) ,r)

w = np.dot(np.linalg.pinv(FI), y_train)

# predykcja

pred = np.zeros([len(y_test)])
for i in range(len(y_test)):
    for j in range(p):
        pred[i] += w[j]*fi(np.sqrt(np.sum((X_test[i] - c[j])**2)) ,r)

error = ((pred - y_test)**2).mean()
print(error)
plt.figure()
t = np.arange(0, len(y_train)+len(y_test), 1)
plt.plot(t[len(t)-backward:len(t)], y[len(y)-backward:len(y)], label='Rzeczywiste')
plt.plot(t[len(t)-days:len(t)], pred[len(pred)-days:len(pred)], 'r', label='Predykcja')
plt.legend()
plt.show()


