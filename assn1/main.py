import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, RidgeClassifier

## This file contains Rough Code for testing some stuffs.

n = 64


def arrtonum(arr, n):
    res = 0
    for i in range(n):
        res += arr[i]*(2**(n-i-1))
    return res


def doshit(arr):
    if arrtonum(arr[64:68], 4) > arrtonum(arr[68:72], 4):
        #print(arr)
        temp = np.array(arr[64:68])
        arr[64:68] = arr[68:72]
        arr[68:72] = temp
        arr[72] = 1- arr[72]
        #print(arr)
        #print("##############################")
    return arr


inp = np.loadtxt("train.dat", dtype=int)
data_trn = np.array([doshit(i) for i in inp])
inp = np.loadtxt("test.dat", dtype=int)
data_tst = np.array([doshit(i) for i in inp])

forta = {}
sorta = {}

for i in data_trn:
    x = arrtonum(i[64:68],4)
    y = arrtonum(i[68:72],4)

    #print("({},{})".format(x,y))

    if (x,y) in forta:
        forta[(x,y)] = np.vstack([forta[(x,y)], np.array([np.append(i[0:64],i[72])])])
    else:
        forta[(x,y)] = np.append(i[0:64],i[72])

for i in data_tst:
    x = arrtonum(i[64:68],4)
    y = arrtonum(i[68:72],4)

    #print("({},{})".format(x,y))

    if (x,y) in sorta:
        sorta[(x,y)] = np.vstack([sorta[(x,y)], np.array([np.append(i[0:64],i[72])])])
    else:
        sorta[(x,y)] = np.append(i[0:64],i[72])


clf = LinearSVC()
#clf = LogisticRegression(random_state=0) 
#clf = RidgeClassifier()
clf.fit(forta[(3,9)][:,:-1], forta[(3,9)][:,-1])
res = clf.predict(sorta[(3,9)][:,:-1])
print(res)
acc = mean_squared_error(sorta[(3,9)][:,-1],res)
print("Error :{} %".format(acc*100))
print(clf.classes_)