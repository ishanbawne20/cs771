import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, RidgeClassifier


class Xorro_Break:
    def __init__(self,no_puf, ques, indsize, tot_len) -> None:
        self.no_puf = no_puf
        self.ques = ques
        self.indsize = indsize
        self.tot_len = tot_len
        pass


    def arrtonum(self, arr, n):
        res = 0
        for i in range(n):
            res += arr[i]*(2**(n-i-1))
        return res


    def doshit(self, arr):
        if self.arrtonum(arr[self.ques:self.ques + self.indsize], self.indsize) > self.arrtonum(arr[self.ques + self.indsize: self.ques + 2*self.indsize], self.indsize):
            temp = np.array(arr[self.ques:self.ques + self.indsize])
            arr[self.ques:self.ques + self.indsize] = arr[self.ques + self.indsize: self.ques + 2*self.indsize]
            arr[self.ques + self.indsize: self.ques + 2*self.indsize] = temp
            arr[self.tot_len-1] = 1 - arr[self.tot_len-1]
        return arr
    
    def prepare(self, inp):
        self.data_trn = np.array([self.doshit(i) for i in inp])
        self.forta = {}

        for i in self.data_trn:
            x = self.arrtonum(i[self.ques:self.ques + self.indsize],self.indsize)
            y = self.arrtonum(i[self.ques + self.indsize: self.ques + 2*self.indsize],self.indsize)

            if (x,y) in self.forta:
                self.forta[(x,y)] = np.vstack([self.forta[(x,y)], np.array([np.append(i[0:self.ques],i[self.tot_len-1])])])
            else:
                self.forta[(x,y)] = np.append(i[0:64],i[self.tot_len-1])
        


    def my_train(self):
        self.models = np.ndarray(shape=(self.no_puf,self.no_puf), dtype=LinearSVC)

        for i in range(self.no_puf):
            for j in range(i+1, self.no_puf):
                self.models[i][j] = LinearSVC().fit(self.forta[(i,j)][:,:-1], self.forta[(i,j)][:,-1])
        pass

    def my_predict(self, tst):
        res = []

        for i in tst:
            x = int(self.arrtonum(i[self.ques:self.ques + self.indsize],self.indsize))
            y = int(self.arrtonum(i[self.ques + self.indsize: self.ques + 2*self.indsize],self.indsize))
            a = np.array([])

            if(x > y):
                a = self.models[y][x].predict(np.array([i[0:self.ques]]))
                a[0] = 1-a[0]
            else:
                a = self.models[x][y].predict(np.array([i[0:self.ques]]))
            res.append(a[0])
        return np.array(res)
    
    def check_error(self, tst, tru_val):
        res = self.my_predict(tst)
        return mean_squared_error(tru_val, res)
