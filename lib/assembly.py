import numpy as np
import numpy.matlib
import multiprocessing as mp
import functools,h5py,time

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from empymod.utils import check_time,conv_warning
from empymod.model import tem
from .mesh import Mesh
from .materials import Domain,Stern,Robin,Dirichlet
from .materials import Consts

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

def assemble_Ke2d(mesh,domain):
    print('Assembling the system of equations for triangular elements')
    print('This will take a while')
    start = time.time()
    n_node = len(mesh.nodes)
    n_elem = len(mesh.elements)
    n_rep = domain.c_x.shape[1]

    I = np.zeros(n_elem*9*n_rep**2,dtype=int)
    J = np.zeros(n_elem*9*n_rep**2,dtype=int)
    V1 = np.zeros(n_elem*9*n_rep**2,dtype=domain.c_x.dtype)
    V2 = np.zeros(n_elem*9*n_rep**2,dtype=domain.c_x.dtype)
    b1 = np.zeros(n_node*n_rep,dtype=domain.c_x.dtype)
    b2 = np.zeros(n_node*n_rep,dtype=domain.c_x.dtype)

    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(3*n_rep,dtype=int),3*n_rep,1).T
    COL = np.matlib.repmat(np.arange(3*n_rep,dtype=int),3*n_rep,1)

    ind_K = np.where(domain.K_stack.ravel())[0]
    ind_b = np.where(domain.b_stack.ravel())[0]

    #check elements to be computed
    mask = np.any(np.sign(domain.a).astype(bool),axis=(1,2))
    mask = mask|np.any(np.sign(domain.c_x).astype(bool),axis=(1,2))
    mask = mask|np.any(np.sign(domain.c_y).astype(bool),axis=(1,2))
    mask = mask|np.any(np.sign(domain.alpha_x).astype(bool),axis=(1,2))
    mask = mask|np.any(np.sign(domain.alpha_y).astype(bool),axis=(1,2))
    mask = mask|np.any(np.sign(domain.beta_x).astype(bool),axis=(1,2))
    mask = mask|np.any(np.sign(domain.beta_y).astype(bool),axis=(1,2))
    
    mask = mask|np.any(np.sign(domain.f).astype(bool),axis=1)
    mask = mask|np.any(np.sign(domain.gamma_x).astype(bool),axis=1)
    mask = mask|np.any(np.sign(domain.gamma_y).astype(bool),axis=1)
    
    #roughly check elements used in the air
    if np.any(np.sign(domain.a_n[mesh.is_on_air]).astype(bool)):
        mask = mask|mesh.is_in_air

    if np.any(np.sign(domain.f_n[mesh.is_on_air]).astype(bool)):
        mask = mask|mesh.is_in_air

    if np.any(np.sign(domain.f_d[mesh.is_on_air]).astype(bool)):
        mask = mask|mesh.is_in_air

    #roughly check elements used in the water
    if np.any(np.sign(domain.a_n[mesh.is_on_water]).astype(bool)):
        mask = mask|mesh.is_in_water
    
    if np.any(np.sign(domain.f_n[mesh.is_on_water]).astype(bool)):
        mask = mask|mesh.is_in_water

    if np.any(np.sign(domain.f_d[mesh.is_on_water]).astype(bool)):
        mask = mask|mesh.is_in_water

    #roughly check elements used in the solid
    if np.any(np.sign(domain.a_n[mesh.is_on_solid]).astype(bool)):
        mask = mask|mesh.is_in_solid
    
    if np.any(np.sign(domain.f_n[mesh.is_on_solid]).astype(bool)):
        mask = mask|mesh.is_in_solid
    
    if np.any(np.sign(domain.f_d[mesh.is_on_water]).astype(bool)):
        mask = mask|mesh.is_in_water

    elem_proc = mask&mesh.is_inside_domain
    #elem_proc = mesh.is_inside_domain
    for i in range(n_elem):
        cnt = i*9*n_rep**2
        ind_n = mesh.elements[i,:]
        if elem_proc[i]==True:
            #Ke1,Ke2,be1 = build_Ke2d(c_x=c_x[i,:,:],c_y=c_y[i,:,:],
            #    alpha_x=alpha_x[i,:,:],alpha_y=alpha_y[i,:,:],
            #    beta_x=beta_x[i,:,:],beta_y=beta_y[i,:,:],
            #    gamma_x=gamma_x[i,:],gamma_y=gamma_y[i,:],
            #    a=a[i,:,:],f=f[i,:],a_n=a_n[ind_n,:,:],f_n=f_n[ind_n,:],
            #    Je=Je[i,:,:],area=area[i]) #wrapped
            Ke1,Ke2,be1,be2 = quick_build_Ke2d(
                c_x=domain.c_x[i,:,:],c_y=domain.c_y[i,:,:],
                alpha_x=domain.alpha_x[i,:,:],alpha_y=domain.alpha_y[i,:,:],
                beta_x=domain.beta_x[i,:,:],beta_y=domain.beta_y[i,:,:],
                gamma_x=domain.gamma_x[i,:],gamma_y=domain.gamma_y[i,:],
                a=domain.a[i,:,:],f=domain.f[i,:],
                a_n=domain.a_n[ind_n,:,:],f_n=domain.f_n[ind_n,:],
                f_d=domain.f_d[ind_n,:],Je=mesh.elem_basis[i,:,:],
                area=mesh.elem_area[i],ind_K=ind_K,ind_b=ind_b
                ) #wrapped

        else:
            Ke1 = np.zeros((3*n_rep,3*n_rep),dtype=float)
            Ke2 = np.zeros((3*n_rep,3*n_rep),dtype=float)
            be1 = np.zeros(3*n_rep,dtype=float)
            be2 = np.zeros(3*n_rep,dtype=float)

        ind_rep = REP[ind_n,:].ravel()
        I[cnt:cnt+9*n_rep**2] = ind_rep[ROW].ravel()
        J[cnt:cnt+9*n_rep**2] = ind_rep[COL].ravel()
        V1[cnt:cnt+9*n_rep**2] = Ke1.ravel()
        V2[cnt:cnt+9*n_rep**2] = Ke2.ravel()
        b1[ind_rep] = b1[ind_rep]+be1
        b2[ind_rep] = b2[ind_rep]+be2

    K1 = csr_matrix((V1,(I,J)),shape=(n_node*n_rep,n_node*n_rep))
    K2 = csr_matrix((V2,(I,J)),shape=(n_node*n_rep,n_node*n_rep))
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return K1,K2,b1,b2


def build_Ke2d(c_x,c_y,alpha_x,alpha_y,beta_x,beta_y,gamma_x,gamma_y,a,f,
               a_n,f_n,Je,area): #wrapped
    n_rep = len(c_x)
    Ke1 = np.zeros((3*n_rep,3*n_rep),dtype=float)
    Ke2 = np.zeros((3*n_rep,3*n_rep),dtype=float)
    be1 = np.zeros(3*n_rep,dtype=float)
    be2 = np.zeros(3*n_rep,dtype=float)

    for i in range(3*n_rep):
        ii = int(i/n_rep) #ii^th node, i = 1,2,3
        kk = int(i)%n_rep #kk^th unknown, j = 1,2,3,4,...,n_rep
        for j in range(3*n_rep):
            jj = int(j/n_rep) #jk^th node, j = 1,2,3
            ll = int(j)%n_rep #ll^th unknown, l = 1,2,3,4,...,n_rep
            delta = 1-np.abs(np.sign(ii-jj))
            #Ke[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+c_y[kk,ll]*Je[ii,2]*Je[jj,2]+
            #           a[kk,ll]*(1+delta)/12.0+
            #           (alpha_x[kk,ll]*Je[ii,1]+alpha_y[kk,ll]*Je[ii,2])/3.0+
            #           (beta_x[kk,ll]*Je[jj,1]+beta_y[kk,ll]*Je[jj,2])/3.0+
            #           a_n[jj,kk,ll]*(1+delta)/12.0)*area #wrapped
            Ke1[i,j] = (
                c_x[kk,ll]*Je[ii,1]*Je[jj,1]+c_y[kk,ll]*Je[ii,2]*Je[jj,2]
                +(alpha_x[kk,ll]*Je[ii,1]+alpha_y[kk,ll]*Je[ii,2])/3.0
                +(beta_x[kk,ll]*Je[jj,1]+beta_y[kk,ll]*Je[jj,2])/3.0
                )*area #wrapped
            Ke2[i,j] = (a[kk,ll]+a_n[jj,kk,ll])*area*(1+delta)/12.0

        #be[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2]+f[kk]/3.0)*area
        be1[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2])*area
        be2[i] = (f[kk]/3.0)*area

        for jj in range(3):
            delta = 1-np.abs(np.sign(ii-jj))
            be2[i] = be2[i]+f_n[jj,kk]*area*(1+delta)/12.0

    return Ke1,Ke2,be1,be2


def quick_build_Ke2d(c_x,c_y,alpha_x,alpha_y,beta_x,beta_y,gamma_x,gamma_y,a,f,
                     a_n,f_n,f_d,Je,area,ind_K,ind_b): #wrapped
    n_rep = len(c_x)
    Ke1 = np.zeros((3*n_rep,3*n_rep),dtype=c_x.dtype)
    Ke2 = np.zeros((3*n_rep,3*n_rep),dtype=c_x.dtype)
    be1 = np.zeros(3*n_rep,dtype=c_x.dtype)
    be2 = np.zeros(3*n_rep,dtype=c_x.dtype)

    for ij in ind_K:
        i = int(ij/(3*n_rep)) #for i in range(3*n_rep)
        ii = int(i/n_rep) #ii^th node, i = 1,2,3
        kk = int(i)%n_rep #kk^th unknown, j = 1,2,3,4,...,n_rep
        
        j = int(ij)%(3*n_rep) #for j in range(3*n_rep)
        jj = int(j/n_rep) #jj^th node, j = 1,2,3
        ll = int(j)%n_rep #ll^th unknown, l = 1,2,3,4,...,n_rep

        delta = 1-np.abs(np.sign(ii-jj))
        #Ke[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+c_y[kk,ll]*Je[ii,2]*Je[jj,2]+
        #           a[kk,ll]*(1+delta)/12.0+
        #           (alpha_x[kk,ll]*Je[ii,1]+alpha_y[kk,ll]*Je[ii,2])/3.0+
        #           (beta_x[kk,ll]*Je[jj,1]+beta_y[kk,ll]*Je[jj,2])/3.0+
        #           a_n[jj,kk,ll]*(1+delta)/12.0)*area #wrapped
        Ke1[i,j] = (
            c_x[kk,ll]*Je[ii,1]*Je[jj,1]+c_y[kk,ll]*Je[ii,2]*Je[jj,2]
            +(alpha_x[kk,ll]*Je[ii,1]+alpha_y[kk,ll]*Je[ii,2])/3.0
            +(beta_x[kk,ll]*Je[jj,1]+beta_y[kk,ll]*Je[jj,2])/3.0
            )*area #wrapped
        Ke2[i,j] = (a[kk,ll]+a_n[jj,kk,ll])*area*(1+delta)/12.0

    for i in ind_b:
        ii = int(i/n_rep) #ii^th node, i = 1,2,3
        kk = int(i)%n_rep #kk^th unknown, j = 1,2,3,4,...,n_rep
        #be[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2]+f[kk]/3.0)*area
        be1[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2])*area
        be2[i] = (f[kk]/3.0)*area
        for jj in range(3):
            delta = 1-np.abs(np.sign(ii-jj))
            be2[i] = be2[i]+f_n[jj,kk]*area*(1+delta)/12.0 #smooth nodal source
            be2[i] = be2[i]+f_d[jj,kk]*delta/2.0/np.pi #non-smooth Dirac source

    return Ke1,Ke2,be1,be2


def assemble_Ke1d(mesh,stern):
    print('Assembling the system of equations for line segments')
    start = time.time()
    n_node = len(mesh.nodes)
    n_edge = len(mesh.edges)
    n_rep = stern.c_x.shape[1]
    
    I = np.zeros(n_edge*4*n_rep**2,dtype=int)
    J = np.zeros(n_edge*4*n_rep**2,dtype=int)
    V1 = np.zeros(n_edge*4*n_rep**2,dtype=stern.c_x.dtype)
    V2 = np.zeros(n_edge*4*n_rep**2,dtype=stern.c_x.dtype)
    b1 = np.zeros(n_node*n_rep,dtype=stern.c_x.dtype)
    b2 = np.zeros(n_node*n_rep,dtype=stern.c_x.dtype)
    
    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1).T
    COL = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1)

    ind_K = np.where(stern.K_stack.ravel())[0]
    ind_b = np.where(stern.b_stack.ravel())[0]
    
    edge_proc = mesh.is_with_stern
    for i in range(n_edge):
        cnt = i*4*n_rep**2
        ind_n = mesh.edges[i,:]
        if edge_proc[i]==True:
            #Ke1,Ke2,be1 = build_Ke1d(c_x=c_x[i,:,:],alpha_x=alpha_x[i,:,:],
            #                         beta_x=beta_x[i,:,:],gamma_x=gamma_x[i,:],
            #                         a=a[i,:,:],f=f[i,:],Je=Je[i,:,:],
            #                         length=length[i]) #wrapped
            Ke1,Ke2,be1,be2 = quick_build_Ke1d(
                c_x=stern.c_x[i,:,:],
                alpha_x=stern.alpha_x[i,:,:],
                beta_x=stern.beta_x[i,:,:],
                gamma_x=stern.gamma_x[i,:],
                a=stern.a[i,:,:],
                f=stern.f[i,:],
                Je=mesh.edge_basis[i,:,:],
                length=mesh.edge_len[i],
                ind_K=ind_K,ind_b=ind_b) #wrapped
        else:
            Ke1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
            Ke2 = np.zeros((2*n_rep,2*n_rep),dtype=float)
            be1 = np.zeros(2*n_rep,dtype=float)
            be2 = np.zeros(2*n_rep,dtype=float)
        
        ind_rep = REP[ind_n,:].ravel()
        I[cnt:cnt+4*n_rep**2] = ind_rep[ROW].ravel()
        J[cnt:cnt+4*n_rep**2] = ind_rep[COL].ravel()
        V1[cnt:cnt+4*n_rep**2] = Ke1.ravel()
        V2[cnt:cnt+4*n_rep**2] = Ke2.ravel()
        b1[ind_rep] = b1[ind_rep]+be1
        b2[ind_rep] = b2[ind_rep]+be2

    K1 = csr_matrix((V1,(I,J)),shape=(n_node*n_rep,n_node*n_rep))
    K2 = csr_matrix((V2,(I,J)),shape=(n_node*n_rep,n_node*n_rep))
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return K1,K2,b1,b2


def build_Ke1d(c_x,alpha_x,beta_x,gamma_x,a,f,Je,length):
    n_rep = len(c_x)
    Ke1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    Ke2 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    be1 = np.zeros(2*n_rep,dtype=float)
    
    for i in range(2*n_rep):
        ii = int(i/n_rep) #ii^th node, i=1,2,3
        kk = int(i)%n_rep #kk^th unknown, j=1,2,3,4,...,n_rep
        for j in range(2*n_rep):
            jj = int(j/n_rep) #jj^th node, j=1,2,3
            ll = int(j)%n_rep #ll^th unknown, l=1,2,3,4,...,n_rep
            delta = 1-np.abs(np.sign(ii-jj))
            #Ke[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+
            #          a[kk,ll]*(1+delta)/6.0+
            #          (alpha_x[kk,ll]*Je[ii,1])/2.0+
            #          (beta_x[kk,ll]*Je[jj,1])/2.0)*length #wrapped
            Ke1[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]
                        +(alpha_x[kk,ll]*Je[ii,1])/2.0
                        +(beta_x[kk,ll]*Je[jj,1])/2.0)*length #wrapped
            Ke2[i,j] = a[kk,ll]*length*(1+delta)/6.0
            
        #be[i] = (gamma_x[kk]*Je[ii,1]+f[kk]/2.0)*length
        be1[i] = (gamma_x[kk]*Je[ii,1])*length
        be2[i] = (f[kk]/2.0)*length

    return Ke1,Ke2,be1,be2


def quick_build_Ke1d(c_x,alpha_x,beta_x,gamma_x,a,f,Je,length,ind_K,ind_b):
    n_rep = len(c_x)
    Ke1 = np.zeros((2*n_rep,2*n_rep),dtype=c_x.dtype)
    Ke2 = np.zeros((2*n_rep,2*n_rep),dtype=c_x.dtype)
    be1 = np.zeros(2*n_rep,dtype=c_x.dtype)
    be2 = np.zeros(2*n_rep,dtype=c_x.dtype)
    
    for ij in ind_K:
        i = int(ij/(2*n_rep)) #for i in range(2*n_rep)
        ii = int(i/n_rep) #ii^th node, i=1,2,3
        kk = int(i)%n_rep #kk^th unknown, j=1,2,3,4,...,n_rep
        
        j = int(ij)%(2*n_rep) #for j in range(2*n_rep)
        jj = int(j/n_rep) #jj^th node, j=1,2,3
        ll = int(j)%n_rep #ll^th unknown, l=1,2,3,4,...,n_rep
        delta = 1-np.abs(np.sign(ii-jj))
        #Ke[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+
        #          a[kk,ll]*(1+delta)/6.0+
        #          (alpha_x[kk,ll]*Je[ii,1])/2.0+
        #          (beta_x[kk,ll]*Je[jj,1])/2.0)*length #wrapped
        Ke1[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]
                    +(alpha_x[kk,ll]*Je[ii,1])/2.0
                    +(beta_x[kk,ll]*Je[jj,1])/2.0)*length #wrapped
        Ke2[i,j] = a[kk,ll]*length*(1+delta)/6.0
    
    for i in ind_b:
        ii = int(i/n_rep) #ii^th node, i=1,2,3
        kk = int(i)%n_rep #kk^th unknown, j=1,2,3,4,...,n_rep
        #be[i] = (gamma_x[kk]*Je[ii,1]+f[kk]/2.0)*length
        be1[i] = (gamma_x[kk]*Je[ii,1])*length
        be2[i] = (f[kk]/2.0)*length

    return Ke1,Ke2,be1,be2


def assemble_Ks2d(mesh,robin):
    print('Incoorprating the boundary condition of the third kind')
    start = time.time()
    n_node = len(mesh.nodes)
    n_edge = len(mesh.edges)
    n_rep = robin.g_s.shape[1]

    I = np.zeros(n_edge*4*n_rep**2,dtype=int)
    J = np.zeros(n_edge*4*n_rep**2,dtype=int)
    V1 = np.zeros(n_edge*4*n_rep**2,dtype=robin.g_s.dtype)
    V2 = np.zeros(n_edge*4*n_rep**2,dtype=robin.g_s.dtype)
    b1 = np.zeros(n_node*n_rep,dtype=robin.g_s.dtype)
    b2 = np.zeros(n_node*n_rep,dtype=robin.g_s.dtype)

    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1).T
    COL = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1)
    
    ind_K = np.where(robin.K_stack.ravel())[0]
    ind_b = np.where(robin.b_stack.ravel())[0]

    edge_proc = mesh.is_with_stern
    for i in range(n_edge):
        cnt = i*4*n_rep**2
        ind_n = mesh.edges[i,:]
        if edge_proc[i]==True:
            #Ks1,bs1 = build_Ks2d(g_s=g_s[i,:],q_s=q_s[i,:,:],length=length[i])
            Ks1,bs1 = quick_build_Ks2d(
                g_s=robin.g_s[i,:],q_s=robin.q_s[i,:,:],
                length=mesh.edge_len[i],ind_K=ind_K,ind_b=ind_b
                ) #wrapped
        else:
            Ks1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
            bs1 = np.zeros(2*n_rep,dtype=float)

        ind_rep = REP[ind_n,:].ravel()
        I[cnt:cnt+4*n_rep**2] = ind_rep[ROW].ravel()
        J[cnt:cnt+4*n_rep**2] = ind_rep[COL].ravel()
        V1[cnt:cnt+4*n_rep**2] = Ks1.ravel()
        b1[ind_rep] = b1[ind_rep]+bs1

    K1 = csr_matrix((V1,(I,J)),shape=(n_node*n_rep,n_node*n_rep))
    K2 = csr_matrix((V2,(I,J)),shape=(n_node*n_rep,n_node*n_rep))
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return K1,K2,b1,b2


def build_Ks2d(g_s,q_s,length):
    n_rep = len(g_s)
    Ks1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    bs1 = np.zeros(2*n_rep,dtype=float)

    for i in range(2*n_rep):
        ii = int(i/n_rep) #ii^th node, i=1,2
        kk = int(i)%n_rep #kkth unknown, j=1,2,3,4,...,n_rep
        for j in range(2*n_rep):
            jj = int(j/n_rep) #jj^th node, j=1,2
            ll = int(j)%n_rep #ll^th unknown, l=1,2,3,4,...,n_rep
            delta = 1-np.abs(np.sign(ii-jj))
            Ks1[i,j] = q_s[kk,ll]*length*(1+delta)/6.0

        bs1[i] = g_s[kk]*length/2.0

    return Ks1,bs1


def quick_build_Ks2d(g_s,q_s,length,ind_K,ind_b):
    n_rep = len(g_s)
    Ks1 = np.zeros((2*n_rep,2*n_rep),dtype=g_s.dtype)
    bs1 = np.zeros(2*n_rep,dtype=g_s.dtype)

    for ij in ind_K:
        i = int(ij/(2*n_rep)) #for i in range(2*n_rep)
        ii = int(i/n_rep) #ii^th node, i=1,2
        kk = int(i)%n_rep #jj^th unknown, j=1,2,3,4,...,n_rep

        j = int(ij)%(2*n_rep) #for j in range(2*n_rep)
        jj = int(j/n_rep) #jj^th node, j=1,2
        ll = int(j)%n_rep #ll^th unknown, l=1,2,3,4,...,n_rep
        delta = 1-np.abs(np.sign(ii-jj))
        Ks1[i,j] = q_s[kk,ll]*length*(1+delta)/6.0

    for i in ind_b:
        ii = int(i/n_rep) #ii^th node, i=1,2
        kk = int(i)%n_rep #jj^th unknown, j=1,2,3,4,...,n_rep
        bs1[i] = g_s[kk]*length/2.0 #uncomment this

    return Ks1,bs1


def set_first_kind_bc(dirichlet,K_in,b_in,verb=1):
    if verb:
        print('Incoorprating the Dirichlet boundary condition')
        start = time.time()

    mask = dirichlet.on_first_kind_bc.ravel()
    s_n = dirichlet.s_n.ravel()

    K = csr_matrix.copy(K_in)
    b = np.zeros_like(b_in)

    b[~mask] = b_in[~mask]-K.dot(s_n)[~mask]
    b[mask] = s_n[mask]

    ind_n = np.where(mask)[0]
    rows = ind_n
    cols = ind_n
    M = csr_matrix(K.shape).tolil()
    M[rows,cols] = 1.0
    K = zero_rows(K,rows)
    K = zero_cols(K,cols)
    K = K+M

    if verb:
        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')
    return K,b


def zero_rows(M,rows):
    diag = sparse.eye(M.shape[0]).tolil()
    for r in rows:
        diag[r,r] = 0
    #diag[rows,rows] = 0
    return diag.dot(M)


def zero_cols(M,cols):
    diag = sparse.eye(M.shape[0]).tolil()
    for c in cols:
        diag[c,c] = 0
    #diag[cols,cols] = 0
    return M.dot(diag)


def solve_system(K,b,verb=1):
    if verb:
        print('Calling sparse linear system solver')
        start = time.time()
    K.eliminate_zeros()
    sol = spsolve(K,b)
    
    if verb:
        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')
    return sol


def solve_stat(domain,stern,robin,dirichlet,mesh,ratio,a_n,f_n):
    #multipler to rows of domain[1].K2
    n_node,n_rep = dirichlet.s_n.shape
    diag_a = np.zeros((n_node,n_rep),dtype=float)
    diag_a[mesh.is_on_water,0] = a_n
    diag_a = sparse.diags(diag_a.ravel())

    #multiplier to rows of domain[1].b2
    diag_f = np.zeros((n_node,n_rep),dtype=float)
    diag_f[mesh.is_on_water,0] = f_n
    diag_f = sparse.diags(diag_f.ravel())

    #compute K and b and solve the sparse linear system
    K = (domain[0].K1+domain[0].K2
         +diag_a.dot(domain[1].K2)
         +stern.K1+stern.K2
         +robin.K1+robin.K2) #wrapped
    b = (domain[0].b1+domain[0].b2
         +diag_f.dot(domain[1].b2)
         +stern.b1+stern.b2
         +robin.b1*ratio+robin.b2) #wrapped
    K,b = set_first_kind_bc(dirichlet,K,b,verb=0)
    sol = np.reshape(solve_system(K,b,verb=0),(n_node,n_rep))

    return sol


def solve_pert(domain,stern,robin,dirichlet,ratio,freq):
    #multipler to rows of domain[0].K2
    n_node,n_rep = dirichlet.s_n.shape
    diag_a = np.ones((n_node,n_rep),dtype=complex)
    diag_a[:,:-2] = 1j*freq*(2*np.pi) #freq -> angular freq
    diag_a = sparse.diags(diag_a.ravel())

    #multipler to cols of stern.K1
    diag_s1 = np.ones((n_node,n_rep),dtype=float)
    diag_s1[:,-2] = ratio
    diag_s1 = sparse.diags(diag_s1.ravel())

    #multipler to rows/cols of stern.K2
    diag_s2 = np.ones((n_node,n_rep),dtype=complex)
    diag_s2[:,-1] = 1j*freq*(2*np.pi) #freq -> angular freq
    diag_s2 = sparse.diags(diag_s2.ravel())
        
    #compute K and b and solve the sparse linear system
    K = (domain[0].K1+diag_a.dot(domain[0].K2)
         +domain[1].K1
         +stern.K1.dot(diag_s1)+stern.K2.dot(diag_s2)
         +robin.K1+robin.K2)
    b = (domain[0].b1+domain[0].b2
         +stern.b1+stern.b2
         +robin.b1+robin.b2)+0j
    K,b = set_first_kind_bc(dirichlet,K,b,verb=0)
    sol = np.reshape(solve_system(K,b,verb=0),(n_node,n_rep))

    return sol


class FEM():
    def __init__(self,*args,**kwargs):
        if args:
            mesh = args[0]
            pde = args[1]
            domain = Domain(mesh,pde)
            stern = Stern(mesh,pde)
            robin = Robin(mesh,pde)
            dirichlet = Dirichlet(mesh,pde)

            domain.K1,domain.K2,domain.b1,domain.b2 = assemble_Ke2d(mesh,domain)
            stern.K1,stern.K2,stern.b1,stern.b2 = assemble_Ke1d(mesh,stern)
            robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)

            self.mesh = mesh
            self.pde = pde
            self.domain = domain
            self.stern = stern
            self.robin = robin
            self.dirichlet = dirichlet

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)

    def append(self,hf_prefix):
        hf = h5py.File(hf_prefix+'.h5','a')
        dataset = hf['fsol']
        dataset[...] = self.fsol
        hf.close()

    def save(self,hf_prefix):
        hf = h5py.File(hf_prefix+'.h5','w')
        self.mesh.save(hf.create_group('mesh'))
        if 'fsol' in self.__dict__.keys():
            hf.create_dataset('fsol',data=self.fsol)
        else:
            hf.create_dataset('sol',data=self.sol)
        hf.close()

    @classmethod
    def load(cls,hf_prefix):
        fem = cls()
        hf = h5py.File(hf_prefix+'.h5','r')
        for attr in hf.keys():
            if type(hf[attr]) is h5py._hl.dataset.Dataset:
                setattr(fem,attr,np.array(hf[attr][:]))
        fem.mesh = Mesh.load(hf['mesh'])
        hf.close()
        return fem

    def solve(self):
        domain = self.domain
        stern = self.stern
        robin = self.robin
        dirichlet = self.dirichlet
        K = domain.K1+domain.K2+stern.K1+stern.K2+robin.K1+robin.K2
        b = domain.b1+domain.b2+stern.b1+stern.b2+robin.b1+robin.b2
        K,b = set_first_kind_bc(dirichlet,K,b)
        self.sol = solve_system(K,b)

    def fsolve(self,sigma):
        mesh = self.mesh
        pde = self.pde
        self.fsol = np.zeros((len(sigma),len(mesh.nodes)),dtype=complex)
        self.save(mesh.prefix)

        domain = self.domain
        stern = self.stern
        robin = self.robin
        dirichlet = self.dirichlet
        for i in tqdm(range(len(sigma))):
            #robin.g_s[:] = sigma[i,:]
            #robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)
            diag_b = sigma[i,:]
            diag_b = sparse.diags(diag_b.ravel())

            K = (domain.K1+domain.K2+stern.K1+stern.K2
                 +robin.K1+robin.K2)
            b = (domain.b1+domain.b2+stern.b1+stern.b2
                 +diag_b.dot(robin.b1+robin.b2))
            K,b = set_first_kind_bc(dirichlet,K,b,verb=0)
            self.fsol[i,:] = solve_system(K,b,verb=0)
            self.append(mesh.prefix)

    def validate(self,ansol_in,interp=False,mask_in=[],xlim=[],ylim=[],
                 is_static=True):
        n_elem = len(self.mesh.elements)
        n_node = len(self.mesh.nodes)
        n_rep = len(self.pde.c_x[list(self.pde.c_x.keys())[0]])
        if interp:
            x = self.mesh.elem_mids[:,0]
            y = self.mesh.elem_mids[:,1]
            rho,phi = self.mesh.to_spherical(is_nodal=False)
            sol = np.zeros((n_elem,n_rep))
            mask_sol = np.zeros((n_elem,n_rep),dtype=bool)
            for i in range(n_rep):
                sol[:,i] = self.mesh.grad2d(self.sol[:,i])[:,0]
                if i<=(n_rep-3):
                    mask_sol[:,i] = self.mesh.is_in_water
                elif i==(n_rep-1):
                    mask_sol[:,i] = self.mesh.is_in_water #to be updated
                else:
                    mask_sol[:,i] = self.mesh.is_inside_domain
                
                if n_rep==1:
                    mask_sol[:,i] = self.mesh.is_inside_domain
        else:
            x = self.mesh.nodes[:,0]
            y = self.mesh.nodes[:,1]
            rho,phi = self.mesh.to_spherical(is_nodal=True)
            phi = phi*180/np.pi
            sol = self.sol
            mask_sol = np.zeros((n_node,n_rep),dtype=bool)
            for i in range(n_rep):
                if i<=(n_rep-3):
                    mask_sol[:,i] = self.mesh.is_on_water
                elif i==(n_rep-1):
                    mask_sol[:,i] = self.mesh.is_on_stern
                else:
                    mask_sol[:,i] = self.mesh.is_on_air|self.mesh.is_on_water
                    mask_sol[:,i] = mask_sol[:,i]|self.mesh.is_on_solid

                if n_rep==1:
                    mask_sol[:,i] = self.mesh.is_on_air|self.mesh.is_on_water
                    mask_sol[:,i] = mask_sol[:,i]|self.mesh.is_on_solid

        if callable(ansol_in):
            ansol = ansol_in(x=x,y=y)
        else:
            ansol = ansol_in

        if len(mask_in)==0:
            mask_in = np.ones_like(x,dtype=bool)

        if n_rep==1:
            pot = sol[:,0]
            anpot = ansol[:,0]
        else:
            pot = sol[:,-2]
            anpot = ansol[:,-2]

        if n_rep==1:
            titles = ['$U_a$']
            units = [' (V)']
        elif n_rep==4:
            titles = ['$C_-}$','$C_+}$','$U_a$','$\Sigma_s$']
            units = [' $(mol/m^3)$',' $(mol/m^3)$',' $(V)$',' $(C/m^2)$']
        else:
            titles = [None]*n_rep
            units = [None]*n_rep
            for i in range(n_rep-2):
                titles[i] = '$C_'+str(i+1)+'$'
                units[i] = ' $(mol/m^3)$'
            titles[-2] = '$U_a$'
            titles[-1] = '$\Sigma_s$'
            units[-2] = ' $(V)$'
            units[-1] = ' $(C/m^2)$'

        if is_static:
            for i in range(n_rep):
                titles[i] = titles[i]+'$^{(0)}$'
        else:
            for i in range(n_rep):
                titles[i] = '$\delta$ '+titles[i]

        if n_rep>1:
            #titles[-1] = '$-\Sigma_d^{(0)}$'
            titles[-1] = '$\Sigma-\Sigma_s^{(0)}$'

        if np.any(np.iscomplex(anpot)):
            self.mesh.tripcolor(np.real(anpot),title='Real($U_{an}$)')
            self.mesh.tripcolor(np.imag(anpot),title='Imag($U_{an}$)')
        else:
            self.mesh.tripcolor(np.real(anpot),title='Real($U_{an}$)')

        dist = np.sqrt(x**2+y**2)
        if n_rep==1:
            fig,ax = plt.subplots(n_rep,2,figsize=(12,4*n_rep))
        else:
            fig,ax = plt.subplots(n_rep+1,2,figsize=(12,4*(n_rep+1)))
        axes = ax.flatten()
        for i in range(n_rep):
            mask = mask_sol[:,i]&mask_in
            axes[2*i].plot(dist[mask],np.real(sol[mask,i]),'.')
            axes[2*i].plot(dist[mask],np.real(ansol[mask,i]),'.',alpha=0.5)
            axes[2*i].plot(dist[mask],np.abs(np.real(ansol[mask,i]
                                                     -sol[mask,i])),
                           '.',alpha=0.5) #wrapped
            axes[2*i+1].plot(dist[mask],np.imag(sol[mask,i]),'.')
            axes[2*i+1].plot(dist[mask],np.imag(ansol[mask,i]),'.',alpha=0.5)
            axes[2*i+1].plot(dist[mask],np.imag(np.real(ansol[mask,i]
                                                        -sol[mask,i])),
                             '.',alpha=0.5) #wrapped
            axes[2*i+1].legend(['numerical','analytical','abs(diff)'],
                               loc='upper right',bbox_to_anchor=(1.75,1.00)) #wrapped
            axes[2*i].set_title('Real')
            axes[2*i+1].set_title('Imag')
            axes[2*i].set_xlabel('Distance (m)')
            axes[2*i+1].set_xlabel('Distance (m)')
            axes[2*i].set_ylabel(titles[i]+units[i])
            axes[2*i+1].set_ylabel(titles[i]+units[i])
            if (i<=(n_rep-2))&(len(xlim)>0):
                axes[2*i].set_xlim(xlim)
                axes[2*i+1].set_xlim(xlim)

        for i in range(n_rep-1,n_rep):
            if n_rep==1:
                break
            mask = mask_sol[:,i]&mask_in
            axes[2*i+2].plot(phi[mask],np.real(sol[mask,i]),'.')
            axes[2*i+2].plot(phi[mask],np.real(ansol[mask,i]),'.',alpha=0.5)
            axes[2*i+2].plot(phi[mask],np.abs(np.real(ansol[mask,i]
                                                     -sol[mask,i])),
                             '.',alpha=0.5) #wrapped
            axes[2*i+3].plot(phi[mask],np.imag(sol[mask,i]),'.')
            axes[2*i+3].plot(phi[mask],np.imag(ansol[mask,i]),'.',alpha=0.5)
            axes[2*i+3].plot(phi[mask],np.imag(np.real(ansol[mask,i]
                                                        -sol[mask,i])),
                             '.',alpha=0.5) #wrapped
            axes[2*i+3].legend(['numerical','analytical','abs(diff)'],
                               loc='upper right',bbox_to_anchor=(1.75,1.00)) #wrapped
            axes[2*i+2].set_title('Real')
            axes[2*i+3].set_title('Imag')
            axes[2*i+2].set_xlabel('$\phi$ (deg)')
            axes[2*i+3].set_xlabel('$\phi$ (deg)')
            axes[2*i+2].set_ylabel(titles[i]+units[i])
            axes[2*i+3].set_ylabel(titles[i]+units[i])

        plt.tight_layout()
        plt.show()
        return fig,ax

    def visualize(self,xlim=[],ylim=[]):
        #only works for n_rep equal 4 now
        n_rep = len(self.pde.c_x[list(self.pde.c_x.keys())[0]])
        if n_rep==1:
            lbl = ['$U^{(0)}$']
        elif n_rep==4:
            lbl = ['$\delta C_-$','$\delta C_+$']
            lbl = lbl+['$\delta U_s$','$\delta \Sigma_s$']
            lbl[1] = '$\Sigma_d$'
        else:
            lbl = ['$\delta C_-{}$'.format(i+1) for i in range(n_rep-2)]
            lbl = lbl+['$\delta U_s$','$\delta \Sigma_s$']

        mask = self.mesh.is_on_stern
        sol = self.sol[0][0]
        fig,ax=plt.subplots(n_rep,4,figsize=(16,n_rep*4),sharex=True)
        axes = ax.flatten()
        for i in range(n_rep):
            if self.mesh.axis_symmetry=='Y':
                x = self.mesh.nodes[mask,0]
                f = sol[mask,i]
                if i==1:
                    height_water = 10e-9
                    f = ((sol[mask,0]-sol[mask,1])*Consts.f)*height_water
                axes[4*i+0].plot(x,np.abs(f),'.',c='tab:blue',markersize=5)
                axes[4*i+1].plot(x,np.angle(f)*180/np.pi,'.',c='tab:blue',markersize=5)

                axes[4*i+0].plot(-x,np.abs(f),'.',c='tab:blue',markersize=5)
                axes[4*i+1].plot(-x,np.angle(f)*180/np.pi,'.',c='tab:blue',markersize=5)
                
                axes[4*i+2].plot(x,np.real(f),'.',c='tab:blue',markersize=5)
                axes[4*i+3].plot(x,np.imag(f),'.',c='tab:blue',markersize=5)

                axes[4*i+2].plot(-x,np.real(f),'.',c='tab:blue',markersize=5)
                axes[4*i+3].plot(-x,np.imag(f),'.',c='tab:blue',markersize=5)

            else:
                pass

            axes[4*i].set_ylabel(lbl[i])
        
        axes[0].set_title('Amplitude')
        axes[1].set_title('Phase')
        axes[2].set_title('Real')
        axes[3].set_title('Imag')

        axes[4*n_rep-4].set_xlabel('X (m)')
        axes[4*n_rep-3].set_xlabel('Y (m)')
        axes[4*n_rep-2].set_xlabel('X (m)')
        axes[4*n_rep-1].set_xlabel('Y (m)')
        if len(xlim)==2:
            axes[0].set_xlim(xlim)

        plt.tight_layout()
        plt.show()

    def plot2(self,x,y,i=0):
        nodes = self.mesh.nodes[self.mesh.is_on_stern,:]
        n_ind = np.argmin((nodes[:,0]-x)**2+(nodes[:,1]-y)**2)
        fsol = self.fsol[:,self.mesh.is_on_stern]
#         tsol = self.tsol[i]

        fig,ax = plt.subplots(2,2,figsize=(8,8))
        axes = ax.flatten()

        axes[0].plot(self.freq,np.real(fsol[:,n_ind]),'.')
#         axes[1].plot(self.freq,np.real(fsol[:,n_ind,-1]),'.')
#         axes[2].plot(self.time,np.real(tsol[:,n_ind,-2]),'.')
#         axes[3].plot(self.time,np.real(tsol[:,n_ind,-1]),'.')

        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        axes[2].set_xscale('log')
        axes[3].set_xscale('log')

        axes[0].set_xlabel('Frequency (Hz)')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[2].set_xlabel('Time (s)')
        axes[3].set_xlabel('Time (s)')
        
        axes[0].set_ylabel('$\delta U_s \;(V)$')
        axes[1].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        axes[2].set_ylabel('$\delta U_s \;(V)$')
        axes[3].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        
        #axes[0].set_ylabel('(V)')
        #axes[1].set_ylabel('(C/m^2)')
        #axes[2].set_ylabel('(V)')
        #axes[3].set_ylabel('(C/m^2)')
        
        #axes[0].set_title('(X,Y) = ({0:.2e},{1:.2e})'.format(x,y))
        #axes[1].set_title('(X,Y) = ({0:.2e},{1:.2e})'.format(x,y))
        
        axes[0].set_title('$\delta U_s$')
        axes[1].set_title('$\delta \Sigma_s$')
        axes[2].set_title('$\delta U_s$')
        axes[3].set_title('$\delta \Sigma_s$')

        plt.tight_layout()
        plt.show()


class StatFEM(FEM):
    def __init__(self,*args,**kwargs):
        if args:
            mesh = args[0]
            pde = args[1]
            pb1,pb2 = pde.decompose()
            d1 = Domain(mesh,pb1)
            d2 = Domain(mesh,pb2)
            stern = Stern(mesh,pde)
            robin = Robin(mesh,pde)
            dirichlet = Dirichlet(mesh,pde)

            d1.K1,d1.K2,d1.b1,d1.b2 = assemble_Ke2d(mesh,d1)
            d2.K1,d2.K2,d2.b1,d2.b2 = assemble_Ke2d(mesh,d2)
            stern.K1,stern.K2,stern.b1,stern.b2 = assemble_Ke1d(mesh,stern)
            robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)

            self.mesh = mesh
            self.pde = pde
            self.domain = [d1,d2]
            self.stern = stern
            self.robin = robin
            self.dirichlet = dirichlet

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)

    def save(self,hf_prefix):
        hf = h5py.File(hf_prefix+'.h5','w')
        hf.create_dataset('ratio',data=self.ratio)
        for i in range(len(self.ratio)):
            hf.create_dataset('ratio:{0:1.2f}'.format(self.ratio[i]),
                              data=self.ssol[i])
        self.mesh.save(hf.create_group('mesh'))
        hf.close()

    @classmethod
    def load(cls,hf_prefix):
        stat = cls()
        hf = h5py.File(hf_prefix+'.h5','r')
        for attr in hf.keys():
            if type(hf[attr]) is h5py._hl.dataset.Dataset:
                setattr(stat,attr,np.array(hf[attr][:]))
        stat.mesh = Mesh.load(hf['mesh'])
        hf.close()
        return stat

    def solve(self,ratio,sigma_init=0.002e-2,max_iter=20):
        sigma_solid = self.pde.g_s['is_with_mixed_bound'][0]
        sol = [None]*len(ratio)

        for i in range(len(ratio)):
            sigma_diffuse = -sigma_solid*(1-ratio[i])
            sigma_i = np.min([np.abs(sigma_init),np.abs(sigma_diffuse)])
            sigma_i *= np.sign(sigma_solid)
            #print('sigma_init',sigma_init,'sigma_solid',sigma_solid)
            
            u_1 = np.zeros(np.sum(self.mesh.is_on_water),dtype=float)
            u_i = np.zeros(np.sum(self.mesh.is_on_water),dtype=float)

            sol[i] = []
            func_stat = functools.partial(solve_stat,self.domain,self.stern,
                                          self.robin,self.dirichlet,self.mesh) #wrapped

            print('Iteratively solving nonlinear PB equation for:')
            print('ratio: {0:.2f}'.format(ratio[i]))
            string = 'sigma_d*(-1) [C/(m*m)] : {0:.2E} & {1:.2E} [init. & targ.]'
            print(string.format(sigma_i,-sigma_diffuse))
            print('This will take a while')
            start = time.time()

            for j in tqdm(range(max_iter)):
                a_n = self.pde.func_a(x=[],y=[],u=u_i)
                f_n = self.pde.func_f(x=[],y=[],u=u_i)

                sol[i].append(func_stat(sigma_i/sigma_solid,a_n,f_n))

                #update u_i
                u_1[:] = u_i
                u_i = sol[i][j][self.mesh.is_on_water,0]

                #update sigma_i
                sigma_1 = sigma_i
                sigma_i = np.min([np.abs(sigma_i*5),np.abs(sigma_diffuse)])
                sigma_i *= np.sign(sigma_solid)

                #compute relative error
                if np.linalg.norm(u_i)>0:
                    rel_error = np.linalg.norm(u_i-u_1)/np.linalg.norm(u_i)
                else:
                    rel_error = 0.0
                
                #check convergence
                if (sigma_1==-sigma_diffuse)&(rel_error<0.05):
                    break

            elapsed = time.time()-start
            print('Final relative error is {0:.2f}'.format(rel_error))
            print('Time elapsed ',elapsed,'sec')
            print('')

        self.ratio = ratio
        self.ssol = sol


class PertFEM(FEM):
    def __init__(self,*args,**kwargs):
        #ratio = [1.0] #be careful it is fixed
        #freq = [3e4] #be careful it is fixed
        #sigma_solid = -0.01 #be careful it is fixed
        if args:
            mesh = args[0]
            pde = args[1]
            pnp1,pnp2 = pde.decompose()
            d1 = Domain(mesh,pnp1)
            d2 = Domain(mesh,pnp2)
            stern = Stern(mesh,pde)
            robin = Robin(mesh,pde)
            dirichlet = Dirichlet(mesh,pde)

            d1.K1,d1.K2,d1.b1,d1.b2 = assemble_Ke2d(mesh,d1)
            #d2.K1,d2.K2,d2.b1,d2.b2 = assemble_Ke2d(mesh,d2)
            stern.K1,stern.K2,stern.b1,stern.b2 = assemble_Ke1d(mesh,stern)
            robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)

            self.mesh = mesh
            self.pde = pde
            self.domain = [d1,d2]
            self.stern = stern
            self.robin = robin
            self.dirichlet = dirichlet

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)

    def append(self,hf_prefix):
        hf = h5py.File(hf_prefix+'.h5','a')
        #hf.create_dataset('ratio',data=self.ratio)
        #hf.create_dataset('freq',data=self.freq)
        for i in range(len(self.ratio)):
            #hf.create_dataset('ratio:{0:1.2f}'.format(self.ratio[i]),
            #                  data=self.fsol[i])
            dataset = hf['ratio:{0:1.2f}'.format(self.ratio[i])]
            dataset[...] = self.fsol[i]
        #self.mesh.save(hf.create_group('mesh'))
        hf.close()

    def save(self,hf_prefix):
        hf = h5py.File(hf_prefix+'.h5','w')
        hf.create_dataset('ratio',data=self.ratio)
        hf.create_dataset('freq',data=self.freq)
        for i in range(len(self.ratio)):
            hf.create_dataset('ratio:{0:1.2f}'.format(self.ratio[i]),
                              data=self.fsol[i])
        self.mesh.save(hf.create_group('mesh'))
        hf.close()

    @classmethod
    def load(cls,hf_prefix):
        pert = cls()
        hf = h5py.File(hf_prefix+'.h5','r')
        for attr in hf.keys():
            if type(hf[attr]) is h5py._hl.dataset.Dataset:
                setattr(pert,attr,np.array(hf[attr][:]))
        pert.mesh = Mesh.load(hf['mesh'])
        hf.close()
        return pert

    def solve(self,ratio,freq,stat=None,n_proc=1):
        mesh = self.mesh
        pde = self.pde
        sol = [None]*len(ratio)

        self.ratio = ratio
        self.freq = freq
        self.fsol = [np.zeros((len(freq),len(mesh.nodes),pde.shape[0]),
                              dtype=complex)]*len(ratio)
        self.save(mesh.prefix)

        #sigma_solid is not used when is_solid_metal is True
        for i in range(len(ratio)):
            if type(stat) is list or stat is None:
                pot = np.zeros((np.sum(mesh.is_in_water),3),dtype=float)
            elif type(stat) is str:
                pot = mesh.grad2d(np.real(np.load(stat)))[mesh.is_in_water,:]
            else:
                pot = mesh.grad2d(stat.sol[i][-1][:,0])[mesh.is_in_water,:]

            for k in range(self.domain[1].c_x.shape[1]-2):
                #c_x = pde.c_x['is_in_water'][k][-2](x,y,pot[:,0],k)
                #c_y = pde.c_y['is_in_water'][k][-2](x,y,pot[:,0],k)
                #alpha_x = pde.alpha_x['is_in_water'][k][k](x,y,pot[:,1],k)
                #alpha_y = pde.alpha_y['is_in_water'][k][k](x,y,pot[:,2],k)
                c_x = pde.func_c(x=[],y=[],pot=pot[:,0],i=k)
                c_y = pde.func_c(x=[],y=[],pot=pot[:,0],i=k)
                alpha_x = pde.func_alpha(x=[],y=[],grad=pot[:,1],i=k)
                alpha_y = pde.func_alpha(x=[],y=[],grad=pot[:,2],i=k)
                self.domain[1].c_x[mesh.is_in_water,k,-2] = c_x
                self.domain[1].c_y[mesh.is_in_water,k,-2] = c_y
                self.domain[1].alpha_x[mesh.is_in_water,k,k] = alpha_x
                self.domain[1].alpha_y[mesh.is_in_water,k,k] = alpha_y

            self.domain[1]._scale_by_rot_factor(mesh)
            self.domain[1]._scale_by_dist_factor(mesh)

            d2 = self.domain[1]
            d2.K1,d2.K2,d2.b1,d2.b2 = assemble_Ke2d(mesh,d2)

            sol[i] = []
            func_pert = functools.partial(solve_pert,self.domain,self.stern,
                                          self.robin,self.dirichlet,ratio[i]) #wrapped

            print('Computing frequency dependent solutions for:')
            print('ratio: {0:.2f}'.format(ratio[i]))
            string = 'freq [Hz] : {0:.2E} - {1:.2E} : {2:d} [min-max; #]'
            print(string.format(freq[0],freq[-1],len(freq)))
            print('This will take a while')
            start = time.time()

            if n_proc == 1:
                for j in tqdm(range(len(freq))):
                    sol[i].append(func_pert(freq[j]))
                    self.fsol[i][j,:,:] = sol[i][-1]
                    self.append(mesh.prefix)

            else:
                pool = mp.Pool(processes=n_proc)
                for result in tqdm(pool.imap_unordered(func_pert,freq),
                                        total=len(freq)):
                    sol[i].append(result)
                    self.fsol[i][j,:,:] = sol[i][-1]
                    self.append(mesh.prefix)

                pool.close()
                pool.join()

            elapsed = time.time()-start
            print('Time elapsed ',elapsed,'sec')
            print('')
            #convert sol[i] from list to ndarray.shape (n_freq,n_node,n_rep)
            sol[i] = np.array(sol[i])

        #self.ratio = ratio
        #self.freq = freq
        #self.fsol = sol
        return sol

    def ftsolve(self,ratio,freqtime,stat=None,signal=None,ft='dlf',ftarg={},
                n_proc=1):
        time,freq,ft,ftarg = self.argft(freqtime,signal,ft,ftarg)
        self.ratio = ratio
        self.freqtime = freqtime
        self.stat = stat
        self.signal = signal

        self.time = time
        self.freq = freq
        self.ft = ft
        self.ftarg = ftarg

        self.solve(ratio,freq,stat,n_proc) #compute self.fsol
        self.transform(self.fsol,freq,time,signal,ft,ftarg) #compute self.tsol

    def argft(self,freqtime,signal=None,ft='dlf',ftarg={}):
        #define more default parameters
        if ft=='dlf' and 'dlf' not in ftarg.keys():
            ftarg = {'dlf': 'key_81_CosSin_2009'}

        #prepare parameters used in fourier transform
        if signal is not None:
            time,freq,ft,ftarg = check_time(freqtime,signal,ft,ftarg,verb=3)
            print('')
        else:
            time = None
            freq = freqtime

        return time,freq,ft,ftarg

    def transform(self,fsol,freq,time,signal=None,ft='dlf',ftarg={}):
        tsol = [None]*len(fsol)
        if signal is not None:
            for i in range(len(fsol)):
                mask = self.mesh.is_on_stern
                fEM = np.reshape(fsol[i][:,mask,:],(len(freq),-1))
                off = np.empty(fEM.shape[1])

                tEM,conv = tem(fEM,off,freq,time,signal,ft,ftarg)
                # In case of QWE/QUAD, print Warning if not converged
                conv_warning(conv, ftarg, 'Fourier', verb=3)
                tsol[i] = np.reshape(tEM,(len(time),np.sum(mask),-1))

        self.time = time
        self.tsol = tsol
        return tsol

#     def transform2(self,fsol,freq,time,signal=None,ft='dlf',ftarg={}):
#         tsol = [None]*len(fsol)
#         if signal is not None:
#             for i in range(len(fsol)):
#                 mask = self.mesh.is_on_stern
#                 fEM = np.reshape(fsol[i][:,mask,:],(len(freq),-1))
#                 off = np.empty(fEM.shape[1])

#                 tEM,conv = tem(fEM,off,freq,time,signal,ft,ftarg)
#                 # In case of QWE/QUAD, print Warning if not converged
#                 conv_warning(conv, ftarg, 'Fourier', verb=3)
#                 tsol[i] = np.reshape(tEM,(len(time),np.sum(mask),-1))

#         self.time = time
#         self.tsol = tsol
#         return tsol

    def animate(self,freq=[],time=[],i=0,xlim=[],ylim=[],xscale='linear',
                yscale='linear'):
        nodes = self.mesh.nodes[self.mesh.is_on_stern,:]
        if yscale=='log':
            fsol = np.abs(self.fsol[i][:,self.mesh.is_on_stern,:])
            tsol = np.abs(self.tsol[i])
        else:
            fsol = self.fsol[i][:,self.mesh.is_on_stern,:]
            tsol = self.tsol[i]
        
        f_ind = [0]*len(freq)
        for j in range(len(freq)):
            f_ind[j] = np.argmin((self.freq-freq[j])**2)
        
        t_ind = [0]*len(time)
        for j in range(len(time)):
            t_ind[j] = np.argmin((self.time-time[j])**2)
        
        fig,ax = plt.subplots(2,2,sharex=False,figsize=(10,8))
        axes = ax.flatten()
        
        labels = []
        for j in range(len(freq)):
            axes[0].plot(nodes[:,0],np.real(fsol[f_ind[j],:,-2]),'.')
            axes[1].plot(nodes[:,0],np.real(fsol[f_ind[j],:,-1]),'.')
            labels.append('$f=%.2e$ Hz'%(freq[j]))
        #axes[1].legend(labels,loc='upper right',bbox_to_anchor=(1.75,0.75))
        axes[0].legend(labels,loc='lower right')
        
        labels = []
        for j in range(len(time)):
            axes[2].plot(nodes[:,0],tsol[t_ind[j],:,-2],'.')
            axes[3].plot(nodes[:,0],tsol[t_ind[j],:,-1],'.')
            labels.append('$t=%.2e$ Hz'%(time[j]))
        axes[2].legend(labels,loc='lower right')

        if len(xlim)==2:
            axes[0].set_xlim(xlim)
            axes[1].set_xlim(xlim)
            axes[2].set_xlim(xlim)
            axes[3].set_xlim(xlim)
        
        if len(ylim)==2:
            axes[0].set_ylim(ylim)
            axes[1].set_ylim(ylim)
            axes[2].set_ylim(ylim)
            axes[3].set_ylim(ylim)
        
        axes[0].set_xscale(xscale)
        axes[1].set_xscale(xscale)
        axes[2].set_xscale(xscale)
        axes[3].set_xscale(xscale)
        
        axes[0].set_yscale(yscale)
        axes[1].set_yscale(yscale)
        axes[2].set_yscale(yscale)
        axes[3].set_yscale(yscale)
        
        axes[0].set_xlabel('X (m)')
        axes[1].set_xlabel('X (m)')
        axes[2].set_xlabel('X (m)')
        axes[3].set_xlabel('X (m)')
        
        axes[0].set_ylabel('$\delta U_s \;(V)$')
        axes[1].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        axes[2].set_ylabel('$\delta U_s \;(V)$')
        axes[3].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')

        #axes[0].set_ylabel('(V)')
        #axes[1].set_ylabel('(C/m^2)')
        #axes[2].set_ylabel('(V)')
        #axes[3].set_ylabel('(C/m^2)')
        
        axes[0].set_title('Frequency Domain')
        axes[1].set_title('Frequency Domain')
        axes[2].set_title('Time Domain')
        axes[3].set_title('Time Domain')
        
        plt.tight_layout()
        plt.show()


    def animate2(self,freq=[],time=[],i=0,xlim=[],ylim=[],xscale='linear',
                yscale='linear'):
        nodes = self.mesh.nodes[self.mesh.is_on_stern,:]
        if yscale=='log':
            fsol = np.abs(self.fsol[i][:,self.mesh.is_on_stern,:])
            tsol = np.abs(self.tsol[i])
        else:
            fsol = self.fsol[i][:,self.mesh.is_on_stern,:]
            tsol = self.tsol[i]
        
        f_ind = [0]*len(freq)
        for j in range(len(freq)):
            f_ind[j] = np.argmin((self.freq-freq[j])**2)
        
        t_ind = [0]*len(time)
        for j in range(len(time)):
            t_ind[j] = np.argmin((self.time-time[j])**2)
        
        fig,ax = plt.subplots(2,2,sharex=False,figsize=(10,8))
        axes = ax.flatten()
        
        labels = []
        for j in range(len(freq)):
            axes[0].plot(nodes[:,0],np.real(fsol[f_ind[j],:,-2]),'.')
            axes[1].plot(nodes[:,0],np.real(fsol[f_ind[j],:,-1]),'.')
            labels.append('$f=%.2e$ Hz'%(freq[j]))
        #axes[1].legend(labels,loc='upper right',bbox_to_anchor=(1.75,0.75))
        axes[0].legend(labels,loc='lower right')
        
        labels = []
        for j in range(len(time)):
            tsol_max = np.max(np.abs(tsol[t_ind[j],:,-1]))
            axes[2].plot(nodes[:,0],tsol[t_ind[j],:,-2],'.')
            axes[3].plot(nodes[:,0],tsol[t_ind[j],:,-1]/tsol_max,'.')
            labels.append('$t=%.2e$ Hz'%(time[j]))
        axes[2].legend(labels,loc='lower right')

        if len(xlim)==2:
            axes[0].set_xlim(xlim)
            axes[1].set_xlim(xlim)
            axes[2].set_xlim(xlim)
            axes[3].set_xlim(xlim)
        
        if len(ylim)==2:
            axes[0].set_ylim(ylim)
            axes[1].set_ylim(ylim)
            axes[2].set_ylim(ylim)
            axes[3].set_ylim(ylim)
        
        axes[0].set_xscale(xscale)
        axes[1].set_xscale(xscale)
        axes[2].set_xscale(xscale)
        axes[3].set_xscale(xscale)
        
        axes[0].set_yscale(yscale)
        axes[1].set_yscale(yscale)
        axes[2].set_yscale(yscale)
        axes[3].set_yscale(yscale)
        
        axes[0].set_xlabel('X (m)')
        axes[1].set_xlabel('X (m)')
        axes[2].set_xlabel('X (m)')
        axes[3].set_xlabel('X (m)')
        
        axes[0].set_ylabel('$\delta U_s \;(V)$')
        axes[1].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        axes[2].set_ylabel('$\delta U_s \;(V)$')
        axes[3].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')

        #axes[0].set_ylabel('(V)')
        #axes[1].set_ylabel('(C/m^2)')
        #axes[2].set_ylabel('(V)')
        #axes[3].set_ylabel('(C/m^2)')
        
        axes[0].set_title('Frequency Domain')
        axes[1].set_title('Frequency Domain')
        axes[2].set_title('Time Domain')
        axes[3].set_title('Time Domain')
        
        plt.tight_layout()
        plt.show()
        
    def plot(self,x,y,i=0):
        nodes = self.mesh.nodes[self.mesh.is_on_stern,:]
        n_ind = np.argmin((nodes[:,0]-x)**2+(nodes[:,1]-y)**2)
        fsol = self.fsol[i][:,self.mesh.is_on_stern,:]
        tsol = self.tsol[i]

        fig,ax = plt.subplots(2,2,figsize=(8,8))
        axes = ax.flatten()

        axes[0].plot(self.freq,np.real(fsol[:,n_ind,-2]),'.')
        axes[1].plot(self.freq,np.real(fsol[:,n_ind,-1]),'.')
        axes[2].plot(self.time,np.real(tsol[:,n_ind,-2]),'.')
        axes[3].plot(self.time,np.real(tsol[:,n_ind,-1]),'.')

        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        axes[2].set_xscale('log')
        axes[3].set_xscale('log')

        axes[0].set_xlabel('Frequency (Hz)')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[2].set_xlabel('Time (s)')
        axes[3].set_xlabel('Time (s)')
        
        axes[0].set_ylabel('$\delta U_s \;(V)$')
        axes[1].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        axes[2].set_ylabel('$\delta U_s \;(V)$')
        axes[3].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        
        #axes[0].set_ylabel('(V)')
        #axes[1].set_ylabel('(C/m^2)')
        #axes[2].set_ylabel('(V)')
        #axes[3].set_ylabel('(C/m^2)')
        
        #axes[0].set_title('(X,Y) = ({0:.2e},{1:.2e})'.format(x,y))
        #axes[1].set_title('(X,Y) = ({0:.2e},{1:.2e})'.format(x,y))
        
        axes[0].set_title('$\delta U_s$')
        axes[1].set_title('$\delta \Sigma_s$')
        axes[2].set_title('$\delta U_s$')
        axes[3].set_title('$\delta \Sigma_s$')

        plt.tight_layout()
        plt.show()

    def plot2(self,x,y,i=0):
        nodes = self.mesh.nodes[self.mesh.is_on_stern,:]
        n_ind = np.argmin((nodes[:,0]-x)**2+(nodes[:,1]-y)**2)
        fsol = self.fsol[i][:,self.mesh.is_on_stern,:]
#         tsol = self.tsol[i]

        fig,ax = plt.subplots(2,2,figsize=(8,8))
        axes = ax.flatten()

        axes[0].plot(self.freq,np.real(fsol[:,n_ind,-2]),'.')
        axes[1].plot(self.freq,np.real(fsol[:,n_ind,-1]),'.')
#         axes[2].plot(self.time,np.real(tsol[:,n_ind,-2]),'.')
#         axes[3].plot(self.time,np.real(tsol[:,n_ind,-1]),'.')

        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        axes[2].set_xscale('log')
        axes[3].set_xscale('log')

        axes[0].set_xlabel('Frequency (Hz)')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[2].set_xlabel('Time (s)')
        axes[3].set_xlabel('Time (s)')
        
        axes[0].set_ylabel('$\delta U_s \;(V)$')
        axes[1].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        axes[2].set_ylabel('$\delta U_s \;(V)$')
        axes[3].set_ylabel('$\delta \Sigma_s \;(C/m^2)$')
        
        #axes[0].set_ylabel('(V)')
        #axes[1].set_ylabel('(C/m^2)')
        #axes[2].set_ylabel('(V)')
        #axes[3].set_ylabel('(C/m^2)')
        
        #axes[0].set_title('(X,Y) = ({0:.2e},{1:.2e})'.format(x,y))
        #axes[1].set_title('(X,Y) = ({0:.2e},{1:.2e})'.format(x,y))
        
        axes[0].set_title('$\delta U_s$')
        axes[1].set_title('$\delta \Sigma_s$')
        axes[2].set_title('$\delta U_s$')
        axes[3].set_title('$\delta \Sigma_s$')

        plt.tight_layout()
        plt.show()


class Dict2Class(object):
    def __init__(self,my_dict):
        for key in my_dict:
            setattr(self,key,my_dict[key])


def Class2Dict(instance):
    my_dict = {}
    for attr in instance.__dict__.keys():
        my_dict[attr] = instance.__dict__[attr]
    return my_dict
