import numpy as np
import pandas as pd

def one_hot(List):
    df=pd.DataFrame(List)
    one_hot_values=pd.get_dummies(df[0])
    one_hot_values=one_hot_values.values.tolist()
    one_hot_values=np.asarray(one_hot_values,dtype=np.float64)
    return one_hot_values