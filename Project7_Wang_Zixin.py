# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:27:09 2018

@author: Zixin
"""

import numpy as np
import matplotlib.pyplot as plt
import functools

#generate boundary condition of PDE

def PX_boundary(T,r,sigma,strike,dx,S0min,S0max): 
    Xmin=np.log(S0min)
    Xmax=np.log(S0max)
    XT1=np.arange(Xmin-dx,min(Xmin+r*T-3*sigma*np.sqrt(T)-dx,Xmin-dx),-dx)
    XT2=np.arange(Xmin+dx,max(Xmax+r*T+3*sigma*np.sqrt(T)+dx,Xmax+dx),dx)
    XT=np.concatenate((XT2[::-1],np.array([Xmin]),XT1))
    ST1=np.exp(XT1)
    ST2=np.exp(XT2)
    ST=np.concatenate((ST2[::-1],np.array([S0min]),ST1))
    
    N=len(XT)
    Pt=np.zeros(N)
    at_the_money=N-int((np.log(strike)-min(Xmin+r*T-3*sigma*np.sqrt(T)-dx,Xmin-dx))/dx)-1
    Pt[at_the_money:]=strike-ST[at_the_money:]    
    return Pt,-ST[N-1]+ST[N-2],ST


    
def PX_EFD(dt,T,r,sigma,strike,dx,S0min,S0max):
    boundary_condition=PX_boundary(T,r,sigma,strike,dx,S0min,S0max)
    PT=boundary_condition[0]
    N=len(PT)
    steps=int(T/dt)    
    
    prob_u=dt*(sigma**2/(2*dx**2)+(r-sigma**2/2)/(2*dx))
    prob_m=1-dt*sigma**2/dx**2-r*dt
    prob_d=dt*(sigma**2/(2*dx**2)-(r-sigma**2/2)/(2*dx))

    A=(np.diag(prob_m+np.zeros(N),k=0)
        +np.diag(prob_u+np.zeros(N-1),k=-1)
        +np.diag(prob_d+np.zeros(N-1),k=1))

    A[0,:3]=[prob_u,prob_m,prob_d]
    A[N-1,N-3:N]=[prob_u,prob_m,prob_d]
    
    B=np.zeros(N)
    B[N-1]=boundary_condition[1]
    
    def backward(A,B,P):
        return A.dot(P)+B
    
    P0=functools.reduce(lambda P,func:func(A,B,P),
                        (PT,)+(backward,)*steps)
    
    S0=boundary_condition[2]
    sup=np.flatnonzero(S0<=S0max)
    inf=np.flatnonzero(S0>=S0min)
    in_range=np.intersect1d(sup,inf)    
    return P0[in_range],S0[in_range]


    
def PX_IFD(dt,T,r,sigma,strike,dx,S0min,S0max):
    boundary_condition=PX_boundary(T,r,sigma,strike,dx,S0min,S0max)
    PT=boundary_condition[0]
    N=len(PT)
    steps=int(T/dt) 
    
    prob_u=-1/2*dt*(sigma**2/(dx**2)+(r-sigma**2/2)/dx)
    prob_m=1+dt*sigma**2/dx**2+r*dt
    prob_d=-1/2*dt*(sigma**2/(dx**2)-(r-sigma**2/2)/dx)
    
    A=(np.diag(prob_m+np.zeros(N),k=0)
        +np.diag(prob_u+np.zeros(N-1),k=-1)
        +np.diag(prob_d+np.zeros(N-1),k=1))

    A[0,:2]=[1,-1]
    A[N-1,N-2:N]=[1,-1]

    B=np.zeros(N)
    B[N-1]=-boundary_condition[1]

    def backward(A,B,P):
        B[1:N-1]=P[1:N-1]
        return (np.linalg.inv(A)).dot(B)

    P0=functools.reduce(lambda P,func:func(A,B,P),
                        (PT,)+(backward,)*steps)

    S0=boundary_condition[2]
    sup=np.flatnonzero(S0<=S0max)
    inf=np.flatnonzero(S0>=S0min)
    in_range=np.intersect1d(sup,inf)    
    return P0[in_range],S0[in_range]

def PX_CN(dt,T,r,sigma,strike,dx,S0min,S0max):
    boundary_condition=PX_boundary(T,r,sigma,strike,dx,S0min,S0max)
    PT=boundary_condition[0]
    N=len(PT)
    steps=int(T/dt) 

    prob_u=-1/4*dt*(sigma**2/(dx**2)+(r-sigma**2/2)/dx)
    prob_m=1+dt*sigma**2/(2*dx**2)+r*dt/2
    prob_d=-1/4*dt*(sigma**2/(dx**2)-(r-sigma**2/2)/dx)

    A=(np.diag(prob_m+np.zeros(N),k=0)
        +np.diag(prob_u+np.zeros(N-1),k=-1)
        +np.diag(prob_d+np.zeros(N-1),k=1))

    A[0,:2]=[1,-1]
    A[N-1,N-2:N]=[1,-1]

    Z=np.zeros(N)
    Z[N-1]=-boundary_condition[1]

    def backward(A,Z,P):
        Z[1:N-1]=-prob_u*P[:N-2]-(prob_m-2)*P[1:N-1]-prob_d*P[2:N]
        return (np.linalg.inv(A)).dot(Z)

    P0=functools.reduce(lambda P,func:func(A,Z,P),
                        (PT,)+(backward,)*steps)

    S0=boundary_condition[2]
    sup=np.flatnonzero(S0<=S0max)
    inf=np.flatnonzero(S0>=S0min)
    in_range=np.intersect1d(sup,inf)  
    return P0[in_range],S0[in_range]





a=PX_IFD(0.002,0.5,0.04,0.2,10,0.2*np.sqrt(0.002*1),1,30)

plt.plot(a[1],a[0])











