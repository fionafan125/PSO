import numpy as np
import math 
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


Dim=2         #dimension
pop_size=20   #population_size
w_it_max=0.9  #inertia weight max 
w_it_min=0.4  #inertia weight min 
c1=2          #acceleration factor
c2=2          #acceleration factor
max_iter=300  #iterations
Lb=-5.12*np.ones(Dim)  #lower bound
Ub=5.12*np.ones(Dim)   #upper bound

#Rastrigin function 2D
def fun(X):
    x1=X[:,0]
    x2=X[:,1]
    obj_val = 20+(x1**2 - 10 * np.cos(2 * np.pi * x1)) + (x2**2 - 10 * np.cos(2 * np.pi * x2)) 
    return obj_val

#Initialize vel and position
pos=np.zeros((pop_size,Dim))
vel=np.zeros((pop_size,Dim))
for i in range(pop_size):
    for j in range(Dim):
        pos[i,j]=Lb[j]+(Ub[j]-Lb[j])*np.random.random()
for i in range(pop_size):
    for j in range(Dim):
        vel[i,j]=np.random.random()


#Calculate target value         
obj_val=fun(pos)

#record pbest
pbestval=obj_val.copy()
pbest=pos.copy()
    
#record gbest
fminval=np.min(obj_val)
index=np.argmin(obj_val)
gbest=pbest[index]

record=[]
##迭代
for i in range(max_iter):     
         
    #Calculate target value 
    obj_val=fun(pos)
       
    #update pbest
    p=np.where(obj_val<pbestval)
    pbest[p]=pos[p] 
    pbestval[p]=obj_val[p]
    
    #update gbest
    fbestval=np.min(pbestval)
    ind=np.argmin(pbestval)
    record.append(fbestval)
    
    
    if(fbestval<=fminval):
        fminvalue=fbestval
        gbest=pbest[ind]
    
    #update vel and position
    w=w_it_max-(i/max_iter)*(w_it_max-w_it_min)
    
    for i in range(pop_size):
        for j in range(Dim):
            vel[i,j]=w*vel[i,j]+c1*np.random.random()*(pbest[i,j]-pos[i,j])+\
                    c2*np.random.random()*(gbest[j]-pos[i,j])        
            
            pos[i,j]=pos[i,j]+vel[i,j]
                
            #check bound
            if(pos[i,j]<Lb[j]):
                pos[i,j]=Lb[j]
            elif (pos[i,j]>Ub[j]):
                pos[i,j]=Ub[j]   
               
print("最佳解:",gbest)
print("最小值:",fbestval)


fig = plt.figure() 
plt.plot([i for i in range(300)],record)
plt.xlabel("iteration")
plt.ylabel("object value")
plt.show()


X = np.linspace(-5.12, 5.12, 100)     
Y = np.linspace(-5.12, 5.12, 100)     
X, Y = np.meshgrid(X, Y) 
Z = 20+(X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))  
fig = plt.figure() 
ax = Axes3D(fig) 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap="nipy_spectral",)
ax.contourf(X,Y,Z,zdir='z',offset=-2)
plt.title("Rastrigin Function")
plt.show()