

"""
Boore article code 
Stochastic simulation of ground motion


Path response


R. D. D. Esfahani

"""


import numpy as np



class path:
    
    def __init__(self,R0 =1 ,R1 = 70,R2 = None,p1 = 0.0 , p2 = None,cq = 2):
        """
        
        R0, R1, R2 , P1, P2 : Geometrical spearding parameters in equation 9 in Boore article
    
        cq: is the seismic velocity used in the determination of Q(f), equation 8 in Boore article

        """

        self.R0 = R0 *1000
        self.R1 = R1 *1000
        self.p1 = p1

        if R2 is not None:
            self.R2 = R2 *1000
            self.p2 = p2

        else:

            self.R2 = 1000
            self.p2 = 1
            
        self.cq = cq
        
        
    def __call__(self, R, f ):
        """
        
        Eq 8 in Boore 
        
        """

        z = self.attenuation(R)*self.Gsprd(R,f)

        return z
    
    
    def attenuation(self, R):
        """
        Eq 9 in Boore
        
        """
        if R <= self.R1:
            z = self.R0 / (R+1e-10)
        elif (R>self.R1 and (self.R2 is None or R<self.R2)):
            z = (self.R0/self.R1) * (self.R1/R)**self.p1
        elif (R>self.R2 or self.R2 is not None):
            z = (self.R0/self.R1)*((self.R1/self.R2)**self.p1) * (self.R2/R)**(self.p2)
        return z
    
    
    def Gsprd(self, R,f):
        """
        Eq 8 
        """
        return np.exp((-np.pi * f * R) /(self.Q(f) * self.cq))
    
    
    def Q(self,f):
        """
        
        Q function for S2 branch in Fig6 Boore 
        
        """
        return 180 * (f)**0.45 
    

if __name__ == "__main__":
    R = 10
    f = np.linspace(.1,100,5000)
    
    pthres =path(R0 =1,R1 = 40,R2 = None,p1 = 0.5 , p2 = 0.5,cq =3500)

    pth = pthres(R*1000,f)
    import matplotlib.pyplot as plt
    
    plt.plot(f,pth)
    plt.show()
    print(pth)
    
