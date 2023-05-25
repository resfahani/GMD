
import numpy as np 
import source as srcres
import sitem as stress
import path as pthress
import matplotlib.pyplot as plt
import pandas as pd

    
    
    
if __name__=="__main__":
    data_all = pd.read_csv('ESM_flatfile_FAS.csv',sep = ';', low_memory=False)

    
    
    FAS_U_labels = [x for x in data_all.columns if 'U_F' in x]
    frequencies  = [float(x[3:].replace("_", ".")) for x in FAS_U_labels]
    f = np.asarray(frequencies)

    

    
    
    d = np.random.uniform(10,80,10000)
    M = np.random.uniform(3,6.5,10000)
    k = np.random.uniform(0.01,0.03,10000)
    sd = np.random.uniform(50,150,10000)

    np.random.shuffle(M)
    np.random.shuffle(d)
    np.random.shuffle(sd)
    np.random.shuffle(k)
    
    
    np.save("Parm_main", [M , d , sd, k])
    
    np.save("Freq_vector", [f])
    
