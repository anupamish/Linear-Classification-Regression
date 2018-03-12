#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2))
    X=data[...,0:2]
    Y=data[...,2]
    predictor=LinearRegression()
    predictor.fit(X,Y)
    print ("Weights:",predictor.coef_)

if __name__ == '__main__':
    main()