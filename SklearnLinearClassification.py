#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score as Accuracy

def main():
    data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2,3))
    x0=np.ones(len(data))
    data=np.c_[x0,data]
    X=data[:,0:4]
    Y=data[:,4]
    p=Perceptron(random_state=29397,n_iter=45665)
    p.fit(X,Y)
    accuracy = Accuracy(p.predict(X),Y)
    print ("Weights:",p.coef_)
    print("Accuracy:",accuracy)

if __name__ == '__main__':
    main()