import numpy as np

def convert_to_float(array):
    new_array = np.zeros(shape=array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
           try:
               new_array[i][j] = float(array[i][j])
           except Exception as e:
               if str(e) == 'could not convert string to float: ':
                   # NB: if there is a missing value, we encode it by False
                   new_array[i][j] = np.NaN
    return(new_array)