#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568
import sys
import numpy as np
import pandas as pd
from numpy import linalg as LA

def main():
    data = np.array(np.loadtxt(sys.argv[1],delimiter=','))
    D = np.delete(data,[2],1)
    D = np.insert(D,[0],1,axis=1)
    w = np.dot(np.dot(LA.inv(np.dot(D.T,D)),D.T),data[:,-1])
    print("Weights:", w)

if __name__ == '__main__':
    main()