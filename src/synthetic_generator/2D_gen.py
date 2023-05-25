
import numpy as np 
import source as srcres
import sitem as stress
import path as pthress
import matplotlib.pyplot as plt
import pandas as pd



"""
Main code for synthetic data simulation 

R. D. D. Esfahani

"""



def spectrum(m , R, k, f, ctf, sd , rho, vel):
    
    
    sr = srcres.source(sd * 1e5, 0.707, 0.55, 2, 1, rho, vel)
    pthres = pthress.path(R0 =1, R1 = 40, R2 = 70, p1 = 0.5 , p2 = 0.5, cq =3500)
    stres = stress.site(rho, vel)
    

    pth = pthres(R*1000,f)
    st = stres(f,k)
    c ,cf = sr(m , f , ctf , gm = 'vel')
    
    return pth*st*c
    
    
    
if __name__=="__main__":

    
    vel = 3500
    rho = 2800
    sd = 150 
    ctf = "ws"
    k = .03
    
    pthres = pthress.path(R0 =1, R1 = 40, R2 = 140, p1 = 0.5 , p2 = 0.5, cq =3500)
    stres = stress.site(rho, vel)
    
    [M, d, sd, k] = np.load("Parm_main.npy", allow_pickle=True)
    [f] = np.load("Freq_vector.npy", allow_pickle=True)

    
    n = 10000
    cdata = []
    
    k = 0.02
    sd = 150
    
    sr = srcres.source(sd * 1e4, 0.707, 0.55, 2, 1, rho, vel)
    st = stres(f,k)


    for i in range(n):
        
        R = d[i]
        m = M[i]
        
        c ,cf = sr(m , f , ctf , gm = 'acc')
        pth = pthres(R*1000,f)
        b = pth*st*c
        
        cdata.append(b)
        if i%1000 ==0:
            
            print("data number =  %d" %i)
    
    
    np.save("FAS_2D",cdata)
    print(len(cdata))
    plt.loglog(f ,b )
    print([R,m])
    plt.show()
    
