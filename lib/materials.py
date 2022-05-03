import numpy as np
import numpy.matlib
import time
from scipy import constants
from scipy import sparse
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


class Consts():
    pass

Consts.e = constants.e #elementary charge
Consts.epsilon_0 = constants.epsilon_0 #vacuum permittivity
Consts.a = constants.N_A #Avogadro constant
Consts.k = constants.k #Boltzmann constant
Consts.f = constants.value('Faraday constant') #Faraday constant


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

    elem_proc = mesh.inside_domain
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
            
            if i==0:
                print('Source')
                print(domain.f_n[ind_n,:])
                print('ind')
                print(ind_K)
                print(ind_b)
                print('Check5/2')
                print(Ke1)
                print(Ke2)
                print(be1)
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

    edge_proc = mesh.with_stern
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


class StaticDomain():
    def __init__(self,mesh,physics,survey):
        n_node = len(mesh.nodes)
        n_elem = len(mesh.elements)
        n_rep = 1

        #declare all attributes first
        self.c_x = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.c_y = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.alpha_x = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.alpha_y = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.beta_x = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.beta_y = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.gamma_x = np.zeros((n_elem,n_rep),dtype=float)
        self.gamma_y = np.zeros((n_elem,n_rep),dtype=float)
        self.a = np.zeros((n_elem,n_rep,n_rep),dtype=float)
        self.f = np.zeros((n_elem,n_rep),dtype=float)
        self.a_n = np.zeros((n_node,n_rep,n_rep),dtype=float) #a on nodes
        self.f_n = np.zeros((n_node,n_rep),dtype=float) #f on nodes
        self.f_d = np.zeros((n_node,n_rep),dtype=float) #f on point sources
        self.K_symbol = [] #placeholder
        self.b_symbol = [] #placeholder
        self.K_stack = [] #placeholder
        self.b_stack = [] #placeholder
        self.K1 = [] #placeholder
        self.K2 = [] #placeholder
        self.b1 = [] #placeholder
        self.b2 = [] #placeholder

        #set domain properties in the air
        self.c_x[mesh.in_air,:,:] = Consts.epsilon_0
        self.c_y[mesh.in_air,:,:] = Consts.epsilon_0

        #set domain properties in the water
        self.c_x[mesh.in_water,:,:] = physics.perm_a
        self.c_y[mesh.in_water,:,:] = physics.perm_a

        #set domain properties in the solid
        self.c_x[mesh.in_solid,:,:] = physics.perm_i
        self.c_y[mesh.in_solid,:,:] = physics.perm_i

        #set point source for electric field
        for i in range(len(survey.f_0)):
            x = survey.f_0[i][0]
            y = survey.f_0[i][1]
            ind_n = np.argmin((mesh.nodes[:,0]-x)**2+(mesh.nodes[:,1]-y)**2)
            self.f_d[ind_n,:] = survey.f_0[i][2]

        #multiply domain attributes by elem_factor
        for i in range(n_rep):
            for j in range(n_rep):
                self.c_x[:,i,j] = self.c_x[:,i,j]*mesh.elem_factor
                self.c_y[:,i,j] = self.c_y[:,i,j]*mesh.elem_factor
                self.alpha_x[:,i,j] = self.alpha_x[:,i,j]*mesh.elem_factor
                self.alpha_y[:,i,j] = self.alpha_y[:,i,j]*mesh.elem_factor
                self.beta_x[:,i,j] = self.beta_x[:,i,j]*mesh.elem_factor
                self.beta_y[:,i,j] = self.beta_y[:,i,j]*mesh.elem_factor
                self.a[:,i,j] = self.a[:,i,j]*mesh.elem_factor
                self.a_n[:,i,j] = self.a_n[:,i,j]*mesh.elem_factor

            self.gamma_x[:,i] = self.gamma_x[:,i]*mesh.elem_factor
            self.gamma_y[:,i] = self.gamma_y[:,i]*mesh.elem_factor
            self.f[:,i] = self.f[:,i]*mesh.elem_factor
            self.f_n[:,i] = self.f_n[:,i]*mesh.node_factor
            self.f_d[:,i] = self.f_d[:,i]*mesh.node_factor

        #scale domain attributes by distance scaling factor
        self.c_x = self.c_x*mesh.dist_factor**2
        self.c_y = self.c_y*mesh.dist_factor**2
        self.alpha_x = self.alpha_x*mesh.dist_factor
        self.alpha_y = self.alpha_y*mesh.dist_factor
        self.beta_x = self.beta_x*mesh.dist_factor
        self.beta_y = self.beta_y*mesh.dist_factor
        self.a = self.a*1.0
        self.a_n = self.a_n*1.0
        
        self.gamma_x = self.gamma_x*mesh.dist_factor
        self.gamma_y = self.gamma_y*mesh.dist_factor
        self.f = self.f*1.0
        self.f_n = self.f_n*1.0
        self.f_d = self.f_d*1.0

        #find sparsity pattern of the domain attributes
        K_symbol = np.any(np.sign(self.c_x).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.c_y).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.alpha_x).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.alpha_y).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.beta_x).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.beta_y).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.a).astype(bool),axis=0)
        
        b_symbol = np.any(np.sign(self.f).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.gamma_x).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.gamma_y).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.f_n).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.f_d).astype(bool),axis=0)
        b_symbol = np.reshape(b_symbol,(-1,1))

        K_stack = [[None]*3 for i in range(3)]
        b_stack = [[None]*1 for i in range(3)]
        for i in range(3):
            for j in range(3):
                K_stack[i][j] = K_symbol
            b_stack[i][0] = b_symbol
        
        self.K_stack = np.asarray(np.bmat(K_stack),dtype=int)
        self.b_stack = np.asarray(np.bmat(b_stack),dtype=int)
        
        self.K_symbol = csr_matrix(K_symbol.astype(int))
        self.b_symbol = csr_matrix(b_symbol.astype(int))
        
        #compute global matrices K1/K2 and vectors b1/b2
        self.K1,self.K2,self.b1,self.b2= assemble_Ke2d(mesh,self)

    def update(self,mesh):
        #find sparsity pattern of the domain attributes
        K_symbol = np.any(np.sign(self.c_x).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.c_y).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.alpha_x).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.alpha_y).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.beta_x).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.beta_y).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.a).astype(bool),axis=0)
        K_symbol = K_symbol|np.any(np.sign(self.a_n).astype(bool),axis=0)
        
        b_symbol = np.any(np.sign(self.f).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.f_n).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.f_d).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.gamma_x).astype(bool),axis=0)
        b_symbol = b_symbol|np.any(np.sign(self.gamma_y).astype(bool),axis=0)
        b_symbol = np.reshape(b_symbol,(-1,1))

        K_stack = [[None]*3 for i in range(3)]
        b_stack = [[None]*1 for i in range(3)]
        for i in range(3):
            for j in range(3):
                K_stack[i][j] = K_symbol
            b_stack[i][0] = b_symbol
        
        self.K_stack = np.asarray(np.bmat(K_stack),dtype=int)
        self.b_stack = np.asarray(np.bmat(b_stack),dtype=int)
        
        self.K_symbol = csr_matrix(K_symbol.astype(int))
        self.b_symbol = csr_matrix(b_symbol.astype(int))
        
        #compute global matrices K1/K2 and vectors b1/b2
        self.K1,self.K2,self.b1,self.b2= assemble_Ke2d(mesh,self)

    def spy(self):
        print('Sparsity pattern for Ke and be (zoom-out vs zoom-in)')
        fig,ax = plt.subplots(1,4,figsize=(8,2))
        axes = ax.flatten()
        axes[0].spy(self.K_stack)
        axes[1].spy(self.b_stack)
        axes[2].spy(self.K_symbol)
        axes[3].spy(self.b_symbol)
        axes[1].set_xticks(range(1))
        axes[3].set_xticks(range(1))
        axes[0].set_xlabel('Ke')
        axes[1].set_xlabel('be')
        axes[2].set_xlabel('Ke')
        axes[3].set_xlabel('be')
        plt.show()


class StaticRobin():
    def __init__(self,mesh,physics,survey):
        sigma_diffuse = 1.0*int(not physics.is_solid_metal)
        n_edge = len(mesh.edges)
        n_rep = 1

        #declare all attributes first
        self.g_s = np.zeros((n_edge,n_rep),dtype=float)
        self.q_s = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.K_symbol = [] #placeholder
        self.b_symbol = [] #placeholder
        self.K_stack = [] #placeholder
        self.b_stack = [] #placeholder
        self.K1 = [] #placeholder
        self.K2 = [] #placeholder
        self.b1 = [] #placeholder
        self.b2 = [] #placeholder

        self.g_s[mesh.with_stern,:] = sigma_diffuse

        #multiply robin attributes by edge_factor
        for i in range(n_rep):
            for j in range(n_rep):
                self.q_s[:,i,j] = self.q_s[:,i,j]*mesh.edge_factor

            self.g_s[:,i] = self.g_s[:,i]*mesh.edge_factor
        
        #scale robin attributes by distance scaling factor
        self.q_s = self.q_s*mesh.dist_factor
        self.g_s = self.g_s*mesh.dist_factor
        
        #find sparsity pattern of robin attributes
        K_symbol = np.any(np.sign(self.q_s).astype(bool),axis=0)
        b_symbol = np.any(np.sign(self.g_s).astype(bool),axis=0)
        b_symbol = np.reshape(b_symbol,(-1,1))

        K_stack = [[None]*2 for i in range(2)]
        b_stack = [[None]*1 for i in range(2)]
        for i in range(2):
            for j in range(2):
                K_stack[i][j] = K_symbol
            b_stack[i][0] = b_symbol
        self.K_stack = np.asarray(np.bmat(K_stack))
        self.b_stack = np.asarray(np.bmat(b_stack))

        self.K_symbol = csr_matrix(K_symbol.astype(int))
        self.b_symbol = csr_matrix(b_symbol.astype(int))
        
        #compute global matrices K1/K2 and vectors b1/b2
        self.K1,self.K2,self.b1,self.b2 = assemble_Ks2d(mesh,self)
    
    def update(self):
        #update the domain attributes by some function
        #placeholder
        
        #compute global matrices K1/K2 and vectors b1/b2
        self.K1,self.K2,self.b1,self.b2 = assemble_Ks2d(mesh,self)

        pass

    def spy(self):
        print('Sparsity pattern for Ks and bs (zoom-out vs zoom-in)')
        fig,ax = plt.subplots(1,4,figsize=(8,2))
        axes = ax.flatten()
        axes[0].spy(self.K_stack)
        axes[1].spy(self.b_stack)
        axes[2].spy(self.K_symbol)
        axes[3].spy(self.b_symbol)
        axes[1].set_xticks(range(1))
        axes[3].set_xticks(range(1))
        axes[0].set_xlabel('Ke')
        axes[1].set_xlabel('be')
        axes[2].set_xlabel('Ke')
        axes[3].set_xlabel('be')
        plt.show()


class StaticDirichlet():
    def __init__(self,mesh,physics,survey):
        n_node = len(mesh.nodes)
        n_rep = 1

        #declare all attributes first
        self.on_first_kind_bc = np.zeros((n_node,n_rep),dtype=bool)
        self.s_n = np.zeros((n_node,n_rep),dtype=float)

        #set dirichlet attributes at infinity
        self.on_first_kind_bc[mesh.on_infinity_nodes,:] = True
        self.s_n[mesh.on_infinity_nodes,:] = 0.0

        #set dirichlet attributes on equipotential surface
        self.on_first_kind_bc[mesh.on_equipotential_nodes,:] = True
        self.s_n[mesh.on_equipotential_nodes,:] = survey.s_0

        #set potential as zero if the solid particle is metal
        if physics.is_solid_metal:
            self.on_first_kind_bc[mesh.on_stern_nodes,:] = True
            self.s_n[mesh.on_stern_nodes,:] = 0.0

        #deactivate unused nodes outside the air, water, and solid
        self.on_first_kind_bc[mesh.on_nodes_outside_domain,:] = True
        self.s_n[mesh.on_nodes_outside_domain,:] = 0.0
        
    def set_1st_kind_bc(self,K_in,b_in):
        print('Incoorprating the Dirichlet boundary condition')
        start = time.time()
        
        on_first_kind_bc = self.on_first_kind_bc.flatten(order='C')
        s_n = self.s_n.flatten(order='C')

        K = csr_matrix.copy(K_in)
        b = np.zeros_like(b_in)

        mask = ~on_first_kind_bc
        b[mask] = b_in[mask]-K.dot(s_n)[mask]
        
        mask = on_first_kind_bc
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
        
    def update(self):
        pass


class PertubStern():
    def __init__(self):
        n_node = len(mesh.nodes)
        n_edge = len(mesh.edges)
        n_rep = 1


class Physics():
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)

        self.C_ion = [val*Consts.a for val in self.c_ion]
        self.Q_ion = [val*Consts.e for val in self.z_ion]
        self.Diff_a = [val*Consts.k*self.temperature/Consts.e
                       for val in self.mu_a] #wrapped
        self.Diff_s = self.mu_s*Consts.k*self.temperature/Consts.e
        self.perm_a = self.rel_perm_a*Consts.epsilon_0
        self.perm_i = self.rel_perm_i*Consts.epsilon_0
        
        n_ion = len(self.c_ion)
        self.lambda_d = [0.0]*n_ion
        for i in range(n_ion):
            dl = np.sqrt(self.perm_a*Consts.k*self.temperature/2
                         /self.Q_ion[i]/self.Q_ion[i]/self.C_ion[i]) #wrapped
            self.lambda_d[i] = dl


class Survey():
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)


