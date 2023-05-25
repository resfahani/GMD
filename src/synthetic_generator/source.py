


"""
Boore article code 
Stochastic simulation of ground motion

source response


D. D. Esfahani


"""

import numpy as np
import matplotlib.pyplot as plt

class source:


    def __init__(self, sd, rp, pf, fsf,RD ,rho, vel):


        self._sd = sd          # Stress drop 
        self._rp = rp          # The radiation pattern, usually averaged over a suitable range of azimuths and take-off angle
        self._pf = pf          # Partition of total shear wave energy into horizontal component (1/sqrt(2))
        self._fsf = fsf        # Free surface effect (taken as 2 in almost all application, which strickly speaking correct just for SH waves)
        self._rho = rho        # The density near the source
        self._vel = vel        # The shear velocity near source
        self._RD =RD           # R



    def __call__(self, m0, frequencies, cft, gm='dis'):
        """
        See Boore (2003) eq. 5.
        """
        
        
        Sa,Sb = self._get_shape(frequencies, m0, cft)
        
        amplitudes = self.Mm(m0) * self._cnst * Sa * Sb * 1e-20
        
        
        if gm == 'dis':
            ma = amplitudes  * ((2*np.pi*frequencies*1j)**0)
        elif gm == 'vel':
            ma = amplitudes  * ((2*np.pi*frequencies*1j)**1)
        elif gm == 'acc':
            ma =  amplitudes * ((2*np.pi*frequencies*1j)**2)


        return abs(ma) , self.cf(m0,cft)

    
    
    def Mm(self,m):
        return  10**(3/2*(m + 10.7))*1





    @property
    def _cnst(self):
        """
        
        Equation 7 in boore article
        RD is a reference distance usually 1 km 
        
        """
        
        c = float((self._rp * self._pf * self._fsf) /
                  (4 * np.pi * self._rho * self._vel**(3) * self._RD)) * 1e9 * 1e3
        
        return c
    
    

    def cf(self, m , cft):
        
        """
        Eq 4 in Boore article
        Corner frequency calculation
        
        f0: corner frequency
        _sd: stress drop in bars to
        _m0: Seismic moment in dyne-cm
        _vel: shear wave velocity near source in Km/s
        """
        
        
        if cft == 'ws':
            fa = 4.9 * 1e6 * (self._sd / self.Mm(m))**(1/3) * self._vel *(1e-5)**(1/3) * 1e-3
            fb = 1
            ee = 1
            
            
        elif cft == 'BC92':
            
            fa = np.where(m>=5.3, 10**(3.409-0.681*m), 10**(2.452-0.5*m))
            fb = np.where(m<5.3, 10**(1.495-0.319*m), 10**(2.452-0.5*m))   
            ee = 1
            
            
        elif cft =="H96":
            fa = 10**(2.3 - 0.5*m)
            fb = 10**(3.4 - 0.5*m)
            ee = 1
        
        
        
        elif cft =="AS00":
            fa = np.where(m>=2.4, 10**(2.181-0.496*m), 10**(1.431-0.5*(m-2.4)))
            fb = np.where(m<2.4, 10**(2.41-0.408*m), 10**(1.431-0.5*(m-2.4)))   
            ee = np.where(m>=2.4, 10**(0.605-0.255*m),0)
        
        
        
        return fa , fb ,ee
    
    
    
    def _get_shape(self, frequencies,m,cft):
    
        """
        Fea 96 table2 in Boore article
    
        """
        
        fa,fb,ee = self.cf(m ,cft)
        
        
        if cft == 'ws':
            Sa = 1 / (1 + (frequencies / fa )**(2))
            Sb = 1
        
        
        elif cft =='BC92':
            Sa = np.where(f>fa,fa/f,1)
            Sb = 1/(1+(frequencies/fb)**(2))**(0.5)
        
        
        elif cft =="H96":
            Sa = 1/(1+(frequencies/fa)**(8))**(1/8)
            Sb = 1/(1+(frequencies/fb)**(8))**(1/8)
        
        
        elif cft =="AS00":
            Sa = (1-ee)/(1+(frequencies/fa)**2) + (ee)/(1+(frequencies/fb)**2)
            Sb = 1
        
        return Sa,Sb
    
if __name__=="__main__":
    
    vel = 3500
    rho = 2800
    sd = 50
    m = 4
    ctf = "ws"
    f = np.linspace(.02,50,100)
    sr = source(sd * 1e5, 0.707, 0.55, 2, 1, rho, vel)
    c ,cf = sr(m , f , ctf , gm = 'vel')
    
    plt.loglog(f,c)
    sd = 200
    ctf = "ws"
    f = np.linspace(.02,50,100)
    sr = source(sd * 1e5, 0.707, 0.55, 2, 1, rho, vel)
    c ,cf = sr(m , f , ctf , gm = 'vel')
    
    plt.loglog(f,c)

    plt.show()
    
    print(cf)
