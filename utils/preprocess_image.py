import numpy as np
from utils.onehot_encoder import one_hot

def preprocess(X,y):
    X=np.asarray(X,dtype=np.float32)
    X/=255
    y=one_hot(y)
    y=np.asarray(y,dtype=np.float32)
    return (X,y)