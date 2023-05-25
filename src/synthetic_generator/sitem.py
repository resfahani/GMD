
"""
Boore article code 
Stochastic simulation of ground motion


Site response


R. D. D. Esfahani
"""

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt



class site:
    
    def __init__(self, rho, vel):
        
        self._vel = vel    # Shear wave velocity near source
        self._rho = rho    # Density near source
        
    def __call__(self, f, k):
        
        
        F,A = self.est()
        F = np.insert(F, 0, 1e-2)
        A = np.insert(A, 0, 1)
        eq1  = interp.interp1d(F, A, 'linear', fill_value='extrapolate')
        At  = eq1(f) * np.exp(-np.pi * k * f )
        
        return At
    
    def est(self):
        ng = 100
        F = np.zeros(ng)
        Zhat = np.zeros(ng)
        v = np.logspace(-5,1,ng)
        
        for i in range(ng):
            zl = np.linspace(0,v[i],100)
            dzl = zl[2]-zl[1]
            vel = self.GRV(zl)
            F[i] = 1/(4 * np.sum (dzl/(vel+1e-20)))
            Zhat[i] = np.sum(self.GD(vel))/(np.sum(1/(vel+1e-20))+1e-20)
        
        A = np.sqrt((self._vel*self._rho)/(Zhat*1e6 + 1e-20))
            
        return F, A
    
    
    
    def GRV(self,d):
        """
        S wave velocity of earth Fig 10 in Boore article
        """


        n = len(d)
        v = np.zeros(n)        

        for i in range(n):
            
            if d[i]<=0.001:
                v[i] = 0.245
            elif (d[i] > 0.001 and d[i] <= 0.03):
                v[i] = 2.206 * d[i] ** (0.272)
            elif (d[i] > 0.03 and d[i] <= 0.19):
                v[i] = 3.542 * d[i] ** (0.407)
            elif (d[i] > 0.19 and d[i] <= 4):
                v[i] = 2.505 * d[i] ** (0.199)
            elif (d[i] > 4 and d[i] <= 8):
                v[i] = 2.927 * d[i] ** (0.086)
                
        return v 
    
    
    def GD(self,beta):         

        return 2.5 + (beta - 0.3)*((2.8-2.5)/(3.5-0.3)) 
    



if __name__ == "__main__":
    vel = 3500
    rho = 2800
    f = np.linspace(.01,60,5000)
    k = .0
    
    
    stres = site(rho,vel)
    st = stres(f,k)
    
    plt.loglog(f,st)
    plt.show()
    print(st)
