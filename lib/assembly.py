import numpy as np
import numpy.matlib
import time

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from .materials import Domain,Stern,Robin,Dirichlet

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
    V1 = np.zeros(n_elem*9*n_rep**2,dtype=float)
    V2 = np.zeros(n_elem*9*n_rep**2,dtype=float)
    b1 = np.zeros(n_node*n_rep,dtype=float)
    b2 = np.zeros(n_node*n_rep,dtype=float)

    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(3*n_rep,dtype=int),3*n_rep,1).T
    COL = np.matlib.repmat(np.arange(3*n_rep,dtype=int),3*n_rep,1)

    ind_K = np.where(domain.K_stack.flatten(order='C'))[0]
    ind_b = np.where(domain.b_stack.flatten(order='C'))[0]

    elem_proc = mesh.is_inside_domain
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
            Ke1,Ke2,be1 = quick_build_Ke2d(
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

        ind_rep = REP[ind_n,:].flatten(order='C')
        I[cnt:cnt+9*n_rep**2] = ind_rep[ROW].flatten(order='C')
        J[cnt:cnt+9*n_rep**2] = ind_rep[COL].flatten(order='C')
        V1[cnt:cnt+9*n_rep**2] = Ke1.flatten(order='C')
        V2[cnt:cnt+9*n_rep**2] = Ke2.flatten(order='C')
        b1[ind_rep] = b1[ind_rep]+be1

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

        be1[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2]+f[kk]/3.0)*area

        for jj in range(3):
            delta = 1-np.abs(np.sign(ii-jj))
            be1[i] = be1[i]+f_n[jj,kk]*area*(1+delta)/12.0

    return Ke1,Ke2,be1


def quick_build_Ke2d(c_x,c_y,alpha_x,alpha_y,beta_x,beta_y,gamma_x,gamma_y,a,f,
                     a_n,f_n,f_d,Je,area,ind_K,ind_b): #wrapped
    n_rep = len(c_x)
    Ke1 = np.zeros((3*n_rep,3*n_rep),dtype=float)
    Ke2 = np.zeros((3*n_rep,3*n_rep),dtype=float)
    be1 = np.zeros(3*n_rep,dtype=float)

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
        be1[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2]+f[kk]/3.0)*area
        for jj in range(3):
            delta = 1-np.abs(np.sign(ii-jj))
            be1[i] = be1[i]+f_n[jj,kk]*area*(1+delta)/12.0 #smooth nodal source
            be1[i] = be1[i]+f_d[jj,kk]*delta/2.0/np.pi #non-smooth Dirac source

    return Ke1,Ke2,be1


def assemble_Ke1d(mesh,stern):
    print('Assembling the system of equations for line segments')
    start = time.time()
    n_node = len(mesh.nodes)
    n_edge = len(mesh.edges)
    n_rep = stern.c_x.shape[1]
    
    I = np.zeros(n_edge*4*n_rep**2,dtype=int)
    J = np.zeros(n_edge*4*n_rep**2,dtype=int)
    V1 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    V2 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    b1 = np.zeros(n_node*n_rep,dtype=float)
    b2 = np.zeros(n_node*n_rep,dtype=float)
    
    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1).T
    COL = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1)

    ind_K = np.where(stern.K_stack.flatten(order='C'))[0]
    ind_b = np.where(stern.b_stack.flatten(order='C'))[0]
    
    edge_proc = mesh.is_with_stern
    for i in range(n_edge):
        cnt = i*4*n_rep**2
        ind_n = mesh.edges[i,:]
        if edge_proc[i]==True:
            #Ke1,Ke2,be1 = build_Ke1d(c_x=c_x[i,:,:],alpha_x=alpha_x[i,:,:],
            #                         beta_x=beta_x[i,:,:],gamma_x=gamma_x[i,:],
            #                         a=a[i,:,:],f=f[i,:],Je=Je[i,:,:],
            #                         length=length[i]) #wrapped
            Ke1,Ke2,be1 = quick_build_Ke1d(
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
        
        ind_rep = REP[ind_n,:].flatten(order='C')
        I[cnt:cnt+4*n_rep**2] = ind_rep[ROW].flatten(order='C')
        J[cnt:cnt+4*n_rep**2] = ind_rep[COL].flatten(order='C')
        V1[cnt:cnt+4*n_rep**2] = Ke1.flatten(order='C')
        V2[cnt:cnt+4*n_rep**2] = Ke2.flatten(order='C')
        b1[ind_rep] = b1[ind_rep]+be1

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
            
        be1[i] = (gamma_x[kk]*Je[ii,1]+f[kk]/2.0)*length

    return Ke1,Ke2,be1


def quick_build_Ke1d(c_x,alpha_x,beta_x,gamma_x,a,f,Je,length,ind_K,ind_b):
    n_rep = len(c_x)
    Ke1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    Ke2 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    be1 = np.zeros(2*n_rep,dtype=float)
    
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
        be1[i] = (gamma_x[kk]*Je[ii,1]+f[kk]/2.0)*length

    return Ke1,Ke2,be1


def assemble_Ks2d(mesh,robin):
    print('Incoorprating the boundary condition of the third kind')
    start = time.time()
    n_node = len(mesh.nodes)
    n_edge = len(mesh.edges)
    n_rep = robin.g_s.shape[1]

    I = np.zeros(n_edge*4*n_rep**2,dtype=int)
    J = np.zeros(n_edge*4*n_rep**2,dtype=int)
    V1 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    V2 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    b1 = np.zeros(n_node*n_rep,dtype=float)
    b2 = np.zeros(n_node*n_rep,dtype=float)

    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1).T
    COL = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1)
    
    ind_K = np.where(robin.K_stack.flatten(order='C'))[0]
    ind_b = np.where(robin.b_stack.flatten(order='C'))[0]

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

        ind_rep = REP[ind_n,:].flatten(order='C')
        I[cnt:cnt+4*n_rep**2] = ind_rep[ROW].flatten(order='C')
        J[cnt:cnt+4*n_rep**2] = ind_rep[COL].flatten(order='C')
        V1[cnt:cnt+4*n_rep**2] = Ks1.flatten(order='C')
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
    Ks1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    bs1 = np.zeros(2*n_rep,dtype=float)

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
        bs1[i] = g_s[kk]*length/2.0

    return Ks1,bs1


def set_first_kind_bc(dirichlet,K_in,b_in):
    print('Incoorprating the Dirichlet boundary condition')
    start = time.time()

    mask = dirichlet.on_first_kind_bc.flatten(order='C')
    s_n = dirichlet.s_n.flatten(order='C')

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


def solve_system(K,b):
    print('Calling sparse linear system solver')
    start = time.time()
    K.eliminate_zeros()
    sol = spsolve(K,b)
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return sol


class FEM():
    def __init__(self,mesh,pde):
        domain = Domain(mesh,pde)
        stern = Stern(mesh,pde)
        robin = Robin(mesh,pde)
        dirichlet = Dirichlet(mesh,pde)
        
        domain.K1,domain.K2,domain.b1,domain.b2 = assemble_Ke2d(mesh,domain)
        stern.K1,stern.K2,stern.b1,stern.b2 = assemble_Ke1d(mesh,stern)
        robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)
        
        K = domain.K1+domain.K2+stern.K1+stern.K2+robin.K1+robin.K2
        b = domain.b1+domain.b2+stern.b1+stern.b2+robin.b1+robin.b2
        K,b = set_first_kind_bc(dirichlet,K,b)

        self.mesh = mesh
        self.pde = pde
        self.domain = domain
        self.stern = stern
        self.robin = robin
        self.dirichlet = dirichlet
        self.sol = solve_system(K,b)
    
class StaticFEM():
    def __init__(self,mesh,pde):
        sigma_diffuse = -0.1 #will add ratio and iteration
        
        domain = Domain(mesh,pde)
        stern = Stern(mesh,pde)
        robin = Robin(mesh,pde)
        dirichlet = Dirichlet(mesh,pde)
        
        domain.K1,domain.K2,domain.b1,domain.b2 = assemble_Ke2d(mesh,domain)
        stern.K1,stern.K2,stern.b1,stern.b2 = assemble_Ke1d(mesh,stern)
        robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)
        
        K = domain.K1+domain.K2+stern.K1+stern.K2+robin.K1+robin.K2
        b = (domain.b1+domain.b2+stern.b1*sigma_diffuse+stern.b2
             +robin.b1+robin.b2)
        K,b = set_first_kind_bc(dirichlet,K,b)
        
        self.mesh = mesh
        self.pde = pde
        self.domain = domain
        self.stern = stern
        self.robin = robin
        self.dirichlet = dirichlet
        self.sol = np.reshape(solve_system(K,b),(len(mesh.nodes),-1))


class PerturbFEM():
    def __init__(self,mesh,pde):
        ratio = [1.0]
        freq = [3e4]
        sigma_solid = -0.01
        n_node = len(mesh.nodes)
        n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])
        
        domain = Domain(mesh,pde)
        stern = Stern(mesh,pde)
        robin = Robin(mesh,pde)
        dirichlet = Dirichlet(mesh,pde)

        domain.K1,domain.K2,domain.b1,domain.b2 = assemble_Ke2d(mesh,domain)
        stern.K1,stern.K2,stern.b1,stern.b2 = assemble_Ke1d(mesh,stern)
        robin.K1,robin.K2,robin.b1,robin.b2 = assemble_Ks2d(mesh,robin)
        
        
        for i in range(len(ratio)):
            #sigma_stern and sigma_diffuse are not used because is_solid_metal
            #is True
            sigma_stern = -ratio[i]*sigma_solid
            sigma_diffuse = -(1.0-ratio[i])*sigma_solid #0 if ratio is 1.0

            #placeholder to update domain with respecto static solution
            #placeholder to update dirichlet to modify boundary condition
            
            for j in range(len(freq)):
                #multipler to rows of domain.K2
                diag = np.ones((n_node,n_rep),dtype=complex)
                diag[:,:-2] = 1j*freq[j]
                diag = sparse.diags(diag.flatten(order='C'))

                #multipler to rows/cols of stern.K1
                diag_s1 = np.ones((n_node,n_rep),dtype=complex)
                diag_s1[:,-2] = sigma_stern
                diag_s1 = sparse.diags(diag_s1.flatten(order='C'))
                
                #multipler to rows/cols of stern.K2
                diag_s2 = np.ones((n_node,n_rep),dtype=complex)
                diag_s2[:,-1] = 1j*freq[j]
                diag_s2 = sparse.diags(diag_s2.flatten(order='C'))
            
                #need to modify K to include sigma_stern and freq[i]
                #need to modify b to include sigma_stern and freq[i]
                K = (domain.K1+diag.dot(domain.K2)
                     +diag_s1.dot(stern.K1)+diag_s2.dot(stern.K2)
                     +robin.K1+robin.K2)
                b = (domain.b1+domain.b2
                     +stern.b1+stern.b2
                     +robin.b1+robin.b2)+0j 
                K,b = set_first_kind_bc(dirichlet,K,b)

        self.mesh = mesh
        self.pde = pde
        self.domain = domain
        self.stern = stern
        self.robin = robin
        self.dirichlet = dirichlet
        self.sol = np.reshape(solve_system(K,b),(len(mesh.nodes),-1))
