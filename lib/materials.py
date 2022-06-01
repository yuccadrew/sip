import copy,functools
import numpy as np
from scipy import constants
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

class Consts():
    pass

Consts.e = constants.e #elementary charge
Consts.p = constants.epsilon_0 #vacuum permittivity
Consts.n = constants.N_A #Avogadro constant
Consts.k = constants.k #Boltzmann constant
Consts.f = constants.value('Faraday constant') #Faraday constant


# Using lambda arguments: expression
flatten_list = lambda irregular_list:[element for item in irregular_list \
                                      for element in flatten_list(item)] \
                   if type(irregular_list) is list else [irregular_list]


def build_a(physics,x,y,u,*args,**kwargs): #generalized PB equation
    n_ion = len(physics.c_ion)
    a = np.zeros_like(np.array(u)).astype(float)
    for i in range(n_ion):
        k = -physics.z_ion[i]*Consts.e/Consts.k/physics.temperature
        a += physics.z_ion[i]*physics.c_ion[i]*np.exp(k*u)*(-k)

    return a*Consts.f


def build_f(physics,x,y,u,*args,**kwargs): #generalized PB equation
    n_ion = len(physics.c_ion)
    f = np.zeros_like(np.array(u)).astype(float)
    for i in range(n_ion):
        k = -physics.z_ion[i]*Consts.e/Consts.k/physics.temperature
        f += physics.z_ion[i]*physics.c_ion[i]*np.exp(k*u)*(1-k*u)

    return f*Consts.f


def build_c(physics,x,y,pot,i):
    c = np.exp(-physics.z_ion[i]*Consts.e*pot/Consts.k/physics.temperature)
    c *= physics.mu_a[i]*physics.z_ion[i]*physics.c_ion[i] #c or C_ion
    return c


def build_alpha(physics,x,y,grad,i):
    alpha = physics.mu_a[i]*physics.z_ion[i]*grad #grad_x or grad_y
    return alpha


def build_s(physics,x,y,u,*args,**kwargs): #fixed potential at infinity
    s = -physics.e_0[0]*x-physics.e_0[1]*y
    return s


class Domain():
    def __init__(self,mesh,pde):
        #combine mesh and pde to determine domain attributes
        self._set_domain(mesh,pde)
        self._scale_by_rot_factor(mesh)
        self._scale_by_dist_factor(mesh)
        self._set_sparsity()

    def _set_domain(self,mesh,pde):
        #combine mesh and pde to determine domain attributes
        n_node = len(mesh.nodes)
        n_elem = len(mesh.elements)
        n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])

        self.c_x = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.c_y = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.alpha_x = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.alpha_y = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.beta_x = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.beta_y = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.gamma_x = np.zeros((n_elem,n_rep),dtype=pde.dtype)
        self.gamma_y = np.zeros((n_elem,n_rep),dtype=pde.dtype)
        self.a = np.zeros((n_elem,n_rep,n_rep),dtype=pde.dtype)
        self.f = np.zeros((n_elem,n_rep),dtype=pde.dtype)
        self.a_n = np.zeros((n_node,n_rep,n_rep),dtype=pde.dtype) #a on nodes
        self.f_n = np.zeros((n_node,n_rep),dtype=pde.dtype) #f on nodes
        self.f_d = np.zeros((n_node,n_rep),dtype=pde.dtype) #f on point sources

        attributes = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                      'gamma_x','gamma_y','a','f','a_n','f_n','f_d']

        keys = ['is_with_stern','is_on_stern']

        for attr in pde.__dict__.keys():
            if attr not in attributes:
                continue

            for ky in pde.__dict__[attr].keys():
                if ky in keys: #this will be set up by class Stern
                    continue

                val = pde.__dict__[attr][ky]
                mask = mesh.__dict__[ky]
                if len(mask)==n_node:
                    x = mesh.nodes[mask,0]
                    y = mesh.nodes[mask,1]
                else:
                    x = mesh.elem_mids[mask,0]
                    y = mesh.elem_mids[mask,1]

                for i in range(len(val)):
                    if type(val[i]) is not list:
                        if type(val[i]) is str:
                            func = pde.__dict__[val[i]]
                            self.__dict__[attr][mask,i] = func(x,y,0,i)
                        elif callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i] = val[i]

                        continue
                    
                    for j in range(len(val[i])):
                        #print(attr,ky,'[',i,',',j,']',val[i][j])
                        if type(val[i][j]) is str:
                            func = pde.__dict__[val[i][j]]
                            self.__dict__[attr][mask,i,j] = func(x,y,0,i)
                        elif callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i,j] = val[i][j]

    def _scale_by_rot_factor(self,mesh,*args):
        if args: #if tuple is not empty
            attributes = args
        else:
            attributes = self.__dict__.keys()

        attr_1 = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                  'a','a_n','q_s'] #ndim equal 3
        attr_2 = ['gamma_x','gamma_y','f','f_n','f_d','g_s','s_n'] #ndim equal 2

        for attr in attributes:
            if attr not in attr_1+attr_2:
                continue

            if self.__dict__[attr].shape[0]==len(mesh.nodes):
                rot_factor = mesh.node_factor
            else:
                rot_factor = mesh.elem_factor

            for i in range(self.__dict__[attr].shape[1]):
                if self.__dict__[attr].ndim<=2:
                    self.__dict__[attr][:,i] *= rot_factor
                    continue
                    
                for j in range(self.__dict__[attr].shape[2]):
                    self.__dict__[attr][:,i,j] *= rot_factor

    def _scale_by_dist_factor(self,mesh,*args):
        if args: #if tuple is not empty
            attributes = args
        else:
            attributes = self.__dict__.keys()

        attr_1 = ['c_x','c_y'] #multiply by dist_factor**2
        attr_2 = ['alpha_x','alpha_y','beta_x','beta_y','gamma_x','gamma_y',
                  'g_s','q_s'] #multiply by dist_factor
        attr_3 = ['a','f','a_n','f_n','f_d'] ##multiply by 1.0

        for attr in attributes:
            if attr not in attr_1+attr_2+attr_3:
                continue
            
            if attr in attr_1:
                self.__dict__[attr] *= mesh.dist_factor**2
            elif attr in attr_2:
                self.__dict__[attr] *= mesh.dist_factor
            else:
                self.__dict__[attr] *= 1.0

    def _set_sparsity(self):
        #find sparsity pattern of the domain attributes
        K_sparse = np.any(np.sign(self.a).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.a_n).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.c_x).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.c_y).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.alpha_x).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.alpha_y).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.beta_x).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.beta_y).astype(bool),axis=0)

        b_sparse = np.any(np.sign(self.f).astype(bool),axis=0)
        b_sparse = b_sparse|np.any(np.sign(self.f_n).astype(bool),axis=0)
        b_sparse = b_sparse|np.any(np.sign(self.f_d).astype(bool),axis=0)
        b_sparse = b_sparse|np.any(np.sign(self.gamma_x).astype(bool),axis=0)
        b_sparse = b_sparse|np.any(np.sign(self.gamma_y).astype(bool),axis=0)

        K_stack = [[None]*3 for i in range(3)]
        b_stack = [[None]*1 for i in range(3)]
        for i in range(3):
            for j in range(3):
                K_stack[i][j] = K_sparse
            b_stack[i][0] = b_sparse

        self.K_stack = np.asarray(np.bmat(K_stack),dtype=int)
        self.b_stack = np.asarray(np.bmat(b_stack),dtype=int)
        
        self.K_sparse = csr_matrix(K_sparse.astype(int))
        self.b_sparse = csr_matrix(b_sparse.astype(int))
    
    def update(self):
        pass

    def spy(self):
        print('Sparsity pattern for Ke and be (zoom-out vs zoom-in)')
        fig,ax = plt.subplots(1,4,figsize=(8,2))
        axes = ax.flatten()
        axes[0].spy(self.K_stack)
        axes[1].spy(self.b_stack)
        axes[2].spy(self.K_sparse)
        axes[3].spy(self.b_sparse)
        axes[1].set_xticks(range(1))
        axes[3].set_xticks(range(1))
        axes[0].set_xlabel('Ke')
        axes[1].set_xlabel('be')
        axes[2].set_xlabel('Ke')
        axes[3].set_xlabel('be')
        plt.show()


class Stern():
    def __init__(self,mesh,pde):
        #combine mesh and pde to determine stern attributes
        self._set_stern(mesh,pde)
        self._scale_by_rot_factor(mesh)
        self._scale_by_dist_factor(mesh)
        self._set_sparsity()
    
    def _set_stern(self,mesh,pde):
        #combine mesh and pde to determine stern attributes
        n_node = len(mesh.nodes)
        n_edge = len(mesh.edges)
        n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])

        self.c_x = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.c_y = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.alpha_x = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.alpha_y = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.beta_x = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.beta_y = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.gamma_x = np.zeros((n_edge,n_rep),dtype=pde.dtype)
        self.gamma_y = np.zeros((n_edge,n_rep),dtype=pde.dtype)
        self.a = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        self.f = np.zeros((n_edge,n_rep),dtype=pde.dtype)
        
        attributes = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                      'gamma_x','gamma_y','a','f']
        
        keys = ['is_with_stern','is_on_stern']

        for attr in pde.__dict__.keys():
            if attr not in attributes:
                continue

            for ky in pde.__dict__[attr].keys():
                if ky not in keys:
                    continue

                val = pde.__dict__[attr][ky]
                
                mask = mesh.__dict__[ky]
                if len(mask)==n_node:
                    x = mesh.nodes[mask,0]
                    y = mesh.nodes[mask,1]
                else:
                    x = mesh.edge_mids[mask,0]
                    y = mesh.edge_mids[mask,1]

                for i in range(len(val)):
                    if type(val[i]) is not list:
                        if type(val[i]) is str:
                            func = pde.__dict__[val[i]]
                            self.__dict__[attr][mask,i] = func(x,y,0,i)
                        elif callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i] = val[i]

                        continue

                    for j in range(len(val[i])):
                        #print(attr,ky,'[',i,',',j,']',val[i][j])
                        if type(val[i][j]) is str:
                            func = pde.__dict__[val[i][j]]
                            self.__dict__[attr][mask,i,j] = func(x,y,0,i)
                        elif callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i,j] = val[i][j]

    def _scale_by_rot_factor(self,mesh,*args):
        if args: #if tuple is not empty
            attributes = args
        else:
            attributes = self.__dict__.keys()
            
        attr_1 = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                  'a','a_n','q_s'] #ndim equal 3
        attr_2 = ['gamma_x','gamma_y','f','f_n','f_d','g_s','s_n'] #ndim equal 2

        for attr in attributes:
            if attr not in attr_1+attr_2:
                continue
            
            if self.__dict__[attr].shape[0]==len(mesh.nodes):
                rot_factor = mesh.node_factor
            else:
                rot_factor = mesh.edge_factor

            for i in range(self.__dict__[attr].shape[1]):
                if self.__dict__[attr].ndim<=2:
                    self.__dict__[attr][:,i] *= rot_factor
                    continue

                for j in range(self.__dict__[attr].shape[2]):
                    self.__dict__[attr][:,i,j] *= rot_factor

    def _scale_by_dist_factor(self,mesh,*args):
        if args: #if tuple is not empty
            attributes = args
        else:
            attributes = self.__dict__.keys()
            
        attr_1 = ['c_x','c_y'] #multiply by dist_factor**2
        attr_2 = ['alpha_x','alpha_y','beta_x','beta_y','gamma_x','gamma_y',
                  'g_s','q_s'] #multiply by dist_factor
        attr_3 = ['a','f','a_n','f_n','f_d'] ##multiply by 1.0

        for attr in attributes:
            if attr not in attr_1+attr_2+attr_3:
                continue

            if attr in attr_1:
                self.__dict__[attr] *= mesh.dist_factor**2
            elif attr in attr_2:
                self.__dict__[attr] *= mesh.dist_factor
            else:
                self.__dict__[attr] *= 1.0

    def _set_sparsity(self):
        #find sparsity pattern of the domain attributes
        K_sparse = np.any(np.sign(self.a).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.c_x).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.c_y).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.alpha_x).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.alpha_y).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.beta_x).astype(bool),axis=0)
        K_sparse = K_sparse|np.any(np.sign(self.beta_y).astype(bool),axis=0)

        b_sparse = np.any(np.sign(self.f).astype(bool),axis=0)
        b_sparse = b_sparse|np.any(np.sign(self.gamma_x).astype(bool),axis=0)
        b_sparse = b_sparse|np.any(np.sign(self.gamma_y).astype(bool),axis=0)

        K_stack = [[None]*2 for i in range(2)]
        b_stack = [[None]*1 for i in range(2)]
        for i in range(2):
            for j in range(2):
                K_stack[i][j] = K_sparse
            b_stack[i][0] = b_sparse

        self.K_stack = np.asarray(np.bmat(K_stack),dtype=int)
        self.b_stack = np.asarray(np.bmat(b_stack),dtype=int)
        
        self.K_sparse = csr_matrix(K_sparse.astype(int))
        self.b_sparse = csr_matrix(b_sparse.astype(int))
    
    def update(self):
        pass
    
    def spy(self):
        print('Sparsity pattern for Ke and be (zoom-out vs zoom-in)')
        fig,ax = plt.subplots(1,4,figsize=(8,2))
        axes = ax.flatten()
        axes[0].spy(self.K_stack)
        axes[1].spy(self.b_stack)
        axes[2].spy(self.K_sparse)
        axes[3].spy(self.b_sparse)
        axes[1].set_xticks(range(1))
        axes[3].set_xticks(range(1))
        axes[0].set_xlabel('Ke')
        axes[1].set_xlabel('be')
        axes[2].set_xlabel('Ke')
        axes[3].set_xlabel('be')
        plt.show()


class Robin():
    def __init__(self,mesh,pde):
        #combine mesh and pde to determin robin attributes
        self._set_robin(mesh,pde)
        self._scale_by_rot_factor(mesh)
        self._scale_by_dist_factor(mesh)
        self._set_sparsity()

    def _set_robin(self,mesh,pde):
        #combine mesh and pde to determine robin attributes
        n_node = len(mesh.nodes)
        n_edge = len(mesh.edges)
        n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])
        
        self.g_s = np.zeros((n_edge,n_rep),dtype=pde.dtype)
        self.q_s = np.zeros((n_edge,n_rep,n_rep),dtype=pde.dtype)
        
        attributes = ['g_s','q_s']

        for attr in pde.__dict__.keys():
            if attr not in attributes:
                continue

            for ky in pde.__dict__[attr].keys():
                val = pde.__dict__[attr][ky]

                mask = mesh.__dict__[ky]
                if len(mask)==n_node:
                    x = mesh.nodes[mask,0]
                    y = mesh.nodes[mask,1]
                else:
                    x = mesh.edge_mids[mask,0]
                    y = mesh.edge_mids[mask,1]

                for i in range(len(val)):
                    if type(val[i]) is not list:
                        if type(val[i]) is str:
                            func = pde.__dict__[val[i]]
                            self.__dict__[attr][mask,i] = func(x,y,0,i)
                        elif callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i] = val[i]
                        continue

                    for j in range(len(val[i])):
                        #print(attr,ky,'[',i,',',j,']',val[i][j])
                        if type(val[i][j]) is str:
                            func = pde.__dict__[val[i][j]]
                            self.__dict__[attr][mask,i,j] = func(x,y,0,i)
                        elif callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i,j] = val[i][j]

    def _scale_by_rot_factor(self,mesh,*args):
        if args: #if tuple is not empty
            attributes = args
        else:
            attributes = self.__dict__.keys()
            
        attr_1 = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                  'a','a_n','q_s'] #ndim equal 3
        attr_2 = ['gamma_x','gamma_y','f','f_n','f_d','g_s','s_n'] #ndim equal 2

        for attr in attributes:
            if attr not in attr_1+attr_2:
                continue
            
            if self.__dict__[attr].shape[0]==len(mesh.nodes):
                rot_factor = mesh.node_factor
            else:
                rot_factor = mesh.edge_factor

            for i in range(self.__dict__[attr].shape[1]):
                if self.__dict__[attr].ndim<=2:
                    self.__dict__[attr][:,i] *= rot_factor
                    continue
                
                for j in range(self.__dict__[attr].shape[2]):
                    self.__dict__[attr][:,i,j] *= rot_factor

    def _scale_by_dist_factor(self,mesh,*args):
        if args: #if tuple is not empty
            attributes = args
        else:
            attributes = self.__dict__.keys()
            
        attr_1 = ['c_x','c_y'] #multiply by dist_factor**2
        attr_2 = ['alpha_x','alpha_y','beta_x','beta_y','gamma_x','gamma_y',
                  'g_s','q_s'] #multiply by dist_factor
        attr_3 = ['a','f','a_n','f_n','f_d'] ##multiply by 1.0

        for attr in attributes:
            if attr not in attr_1+attr_2+attr_3:
                continue
            
            if attr in attr_1:
                self.__dict__[attr] *= mesh.dist_factor**2
            elif attr in attr_2:
                self.__dict__[attr] *= mesh.dist_factor
            else:
                self.__dict__[attr] *= 1.0
        
    def _set_sparsity(self):
        #find sparsity pattern of the domain attributes
        K_sparse = np.any(np.sign(self.q_s).astype(bool),axis=0)
        b_sparse = np.any(np.sign(self.g_s).astype(bool),axis=0)

        K_stack = [[None]*2 for i in range(2)]
        b_stack = [[None]*1 for i in range(2)]
        for i in range(2):
            for j in range(2):
                K_stack[i][j] = K_sparse
            b_stack[i][0] = b_sparse

        self.K_stack = np.asarray(np.bmat(K_stack),dtype=int)
        self.b_stack = np.asarray(np.bmat(b_stack),dtype=int)
        
        self.K_sparse = csr_matrix(K_sparse.astype(int))
        self.b_sparse = csr_matrix(b_sparse.astype(int))
    
    def update(self):
        pass
    
    def spy(self):
        print('Sparsity pattern for Ks and bs (zoom-out vs zoom-in)')
        fig,ax = plt.subplots(1,4,figsize=(8,2))
        axes = ax.flatten()
        axes[0].spy(self.K_stack)
        axes[1].spy(self.b_stack)
        axes[2].spy(self.K_sparse)
        axes[3].spy(self.b_sparse)
        axes[1].set_xticks(range(1))
        axes[3].set_xticks(range(1))
        axes[0].set_xlabel('Ks')
        axes[1].set_xlabel('bs')
        axes[2].set_xlabel('Ks')
        axes[3].set_xlabel('bs')
        plt.show()


class Dirichlet():
    def __init__(self,mesh,pde):
        #combine mesh and pde to determine dirichlet attributes
        self._set_dirichlet(mesh,pde)

    def _set_dirichlet(self,mesh,pde):
        #combine mesh and pde to determine dirichlet attributes
        n_node = len(mesh.nodes)
        n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])

        #declare all attributes first
        self.on_first_kind_bc = np.zeros((n_node,n_rep),dtype=bool)
        self.s_n = np.zeros((n_node,n_rep),dtype=pde.dtype)

        #update dirichlet attributes accordingly
        attributes = ['s_n']
        
        for attr in pde.__dict__.keys():
            if attr not in attributes:
                continue

            for ky in pde.__dict__[attr].keys():
                val = pde.__dict__[attr][ky]
                mask = mesh.__dict__[ky]
                x = mesh.nodes[mask,0]
                y = mesh.nodes[mask,1]

                for i in range(len(val)):
                    if type(val[i]) is not list:
                        if val[i] == None:
                            continue

                        self.on_first_kind_bc[mask,i] = True

                        if type(val[i]) is str:
                            func = pde.__dict__[val[i]]
                            self.__dict__[attr][mask,i] = func(x,y,0,i)
                        elif callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i] = val[i]

                        continue

                    for j in range(len(val[i])):
                        if val[i][j] == None:
                            continue

                        #print(attr,ky,'[',i,',',j,']',val_i[j])
                        self.on_first_kind_bc[mask,i,j] = True

                        if type(val[i][j]) is str:
                            func = pde.__dict__[val[i][j]]
                            self.__dict__[attr][mask,i,j] = func(x,y,0,i)
                        elif callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0,i)
                        else:
                            self.__dict__[attr][mask,i,j] = val[i][j]

#     def set_first_kind_bc(self,K_in,b_in):
#         print('Incoorprating the Dirichlet boundary condition')
#         start = time.time()

#         mask = self.on_first_kind_bc.flatten(order='C')
#         s_n = self.s_n.flatten(order='C')

#         K = csr_matrix.copy(K_in)
#         b = np.zeros_like(b_in)

#         b[~mask] = b_in[~mask]-K.dot(s_n)[~mask]
#         b[mask] = s_n[mask]
        
#         ind_n = np.where(mask)[0]
#         rows = ind_n
#         cols = ind_n
#         M = csr_matrix(K.shape).tolil()
#         M[rows,cols] = 1.0
#         K = zero_rows(K,rows)
#         K = zero_cols(K,cols)
#         K = K+M

#         elapsed = time.time()-start
#         print('Time elapsed ',elapsed,'sec')
#         print('')
#         return K,b

#     def update(self,*args): #placeholder
#         if args:
#             attributes = args
#         else:
#             attributes = self.__dict__.keys()

#         #combine mesh and pde to determine dirichlet attributes
#         n_node = len(mesh.nodes)
#         n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])

#         #declare all attributes first
#         self.on_first_kind_bc = np.zeros((n_node,n_rep),dtype=bool)
#         self.s_n = np.zeros((n_node,n_rep),dtype=pde.dtype)

#         #update dirichlet attributes accordingly
#         attr_1 = ['s_n']
        
#         for attr in attributes:
#             if attr not in attr_1:
#                 continue

#             for ky in pde.__dict__[attr].keys():
#                 val = pde.__dict__[attr][ky]
#                 mask = mesh.__dict__[ky]
#                 x = mesh.nodes[mask,0]
#                 y = mesh.nodes[mask,1]

#                 for i in range(len(val)):
#                     if type(val[i]) is not list:
#                         if val[i] == None:
#                             continue

#                         self.on_first_kind_bc[mask,i] = True

#                         if callable(val[i]):
#                             self.__dict__[attr][mask,i] = val[i](x,y,0,i)
#                         else:
#                             self.__dict__[attr][mask,i] = val[i]

#                         continue

#                     for j in range(len(val[i])):
#                         if val[i][j] == None:
#                             continue

#                         #print(attr,ky,'[',i,',',j,']',val_i[j])
#                         self.on_first_kind_bc[mask,i,j] = True

#                         if callable(val[i][j]):
#                             self.__dict__[attr][mask,i,j] = val[i][j](x,y,0,i)
#                         else:
#                             self.__dict__[attr][mask,i,j] = val[i][j]

    def update(self):
        pass

    def visualize(self,mesh,pde):
#         mesh = self.mesh
#         pde = self.pde
        n_rep = len(pde.c_x[list(pde.c_x.keys())[0]])
        
        fig,ax = plt.subplots(n_rep,4,figsize=(20,n_rep*4),
                              sharex=True,sharey=True)
        axes = ax.flatten()
        for i in range(n_rep):
            x_n = mesh.nodes[:,0]
            y_n = mesh.nodes[:,1]
            mask = self.on_first_kind_bc[:,i]
            
            #plot self.on_first_kind_bc
            f_n = self.on_first_kind_bc[:,i]
            f = mesh.grad2d(f_n)[:,0]
            
            sc = axes[i*n_rep+0].scatter(x_n[mask],y_n[mask],s=20,c=f_n[mask],
                                   vmin=0,vmax=1,cmap='coolwarm') #wrapped
            fig.colorbar(sc,ax=axes[i*n_rep+0],location='right')
            axes[i*n_rep+0].set_aspect('equal')
            
            tpc = axes[i*n_rep+1].tripcolor(x_n,y_n,mesh.elements,facecolors=f,
                                        edgecolor='none',vmin=0,vmax=1,
                                        cmap='coolwarm') #wrapped
            fig.colorbar(tpc,ax=axes[i*n_rep+1],location='right')
            axes[i*n_rep+1].set_aspect('equal')
            
            #plot self.s_n
            f_n = self.s_n[:,i]
            f = mesh.grad2d(f_n)[:,0]
            
            sc = axes[i*n_rep+2].scatter(x_n[mask],y_n[mask],s=20,c=f_n[mask],
                                     vmin=min(f),vmax=max(f)) #wrapped
            fig.colorbar(sc,ax=axes[i*n_rep+2],location='right')
            axes[i*n_rep+2].set_aspect('equal')
            
            tpc = axes[i*n_rep+3].tripcolor(x_n,y_n,mesh.elements,facecolors=f,
                                        edgecolor='none',vmin=min(f),vmax=max(f),
                                        cmap='viridis') #wrapped
            fig.colorbar(tpc,ax=axes[i*n_rep+3],location='right')
            axes[i*n_rep+3].set_aspect('equal')
            
            axes[n_rep*(n_rep-1)+i].set_xlabel('X (m)')
            axes[i*n_rep].set_ylabel('Y (m)')
            axes[i*n_rep].set_title('$\Gamma_d$')
            axes[i*n_rep+1].set_title('$\Gamma_d$')
            axes[i*n_rep+2].set_title('$s_n$')
            axes[i*n_rep+3].set_title('$s_n$')
        
        plt.tight_layout()
        plt.show()


class PDE():
    def __init__(self,**kwargs): #avoid long list of inputs
        self.dtype = None
        self.shape = None
        for key,value in kwargs.items():
            setattr(self,key,value)
            #print(key,value,type(value))
            if type(value) is dict:
                for ky in value.keys():
                    if type(value[ky]) is list:
                        #print(ky,flatten_list(value[ky]))
                        for val in flatten_list(value[ky]):
                            if type(val) is complex:
                                self.dtype = complex

                        if self.shape is None:
                            self.shape = (len(value),len(value))

        if self.dtype is None:
            self.dtype = float
        
        if self.shape is None:
            self.shape = (0,0)
    
    def _set_dtype(self):
        self.dtype = float
        for attr in self.__dict__.keys():
            #print(attr,self.__dict__[attr].keys())
            if type(self.__dict__[attr]) is dict:
                for ky in self.__dict__[attr].keys():
                    if type(self.__dict__[attr][ky]) is list:
                        #print(attr,ky,self.__dict__[attr][ky])
                        for val in flatten_list(self.__dict__[attr][ky]):
                            if type(val) is complex:
                                self.dtype = complex
                                return

    def _set_shape(self):
        self.shape = (0,0)
        for attr in self.__dict__.keys():
            if type(self.__dict__[attr]) is dict:
                #print(attr,self.__dict__[attr].keys())
                for ky in self.__dict__[attr].keys():
                    if type(self.__dict__[attr][ky]) is list:
                        #print(attr,ky,self.__dict__[attr][ky])
                        val = self.__dict__[attr][ky]
                        self.shape = (len(val),len(val))
                        return

    def visualize(self,*args):
        if args:
            attributes = args
        else:
            attributes = self.__dict__.keys()
        
        attr_1 = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                      'a','a_n','q_s'] #ndim = 3
        attr_2 = ['gamma_x','gamma_y','f','f_d','f_n','g_s','s_n'] #ndim = 2

        for attr in attributes:
            if attr not in attr_1+attr_2:
                continue

            for ky in self.__dict__[attr].keys():
                print('{}[\'{}\']: '.format(attr,ky))
                val = self.__dict__[attr][ky]

                for i in range(len(val)):
                    print('[',end='')
                    if type(val[i]) is not list:
                        if type(val[i]) is str:
                            print('{0:>8}()'.format(val[i]),end='')
                        elif type(val[i]) is functools.partial:
                            print('{0:>8}()'.format(val[i].func.__name__),
                                  end='') #wrapped
                        elif callable(val[i]):
                            print('{0:>8}()'.format(val[i].__name__),end='')
                        elif val[i] is None:
                            print('{0:>10}'.format('None'),end='')
                        else:
                            print('{0:10.2E}'.format(val[i]),end='')

                        print(']')
                        continue

                    for j in range(len(val[i])):
                        #print(val[i][j])
                        if type(val[i][j]) is str:
                            print('{0:>8}()'.format(val[i][j]),end='')
                        elif type(val[i][j]) is functools.partial:
                            print('{0:>14}()'.format(val[i][j].func.__name__),
                                  end='') #wrapped
                        elif callable(val[i][j]):
                            print('{0:>14}()'.format(val[i][j].__name__),end='')
                        elif val[i][j] is None:
                            print('{0:>16}'.format('None'),end='')
                        else:
                            print('{0:16.2E}'.format(val[i][j]),end='')

                    print(']')
                print('')


class StaticPDE(PDE):
    def __init__(self,physics): #avoid long list of inputs
        attributes = ['c_x','c_y','a','f','s_n','q_s']
        for attr in attributes:
            self.__dict__[attr] = {}

        self.func_a = functools.partial(build_a,physics)
        self.func_f = functools.partial(build_f,physics)

        self._set_static_air(physics)
        self._set_static_water(physics)
        self._set_static_solid(physics)
        self._set_static_stern_bound(physics) #Stern equation
        self._set_static_mixed_bound(physics) #Robin B.C.
        self._set_static_inner_bound(physics) #Dirichlet B.C.
        self._set_static_metal_bound(physics) #Dirichlet B.C.
        self._set_static_outer_bound(physics) #Dirichlet B.C.
        self._set_static_unused_nodes(physics) #Dirichlet B.C.
        self._set_dtype()
        self._set_shape()

#     @staticmethod
#     def _build_a(physics):
#         #global build_a
#         def build_a(x,y,u): #generalized PB equation
#             n_ion = len(physics.c_ion)
#             a = np.zeros_like(x)
#             for i in range(n_ion):
#                 k = -physics.q_ion[i]/Consts.k/physics.temperature
#                 a += physics.q_ion[i]*physics.C_ion[i]*np.exp(k*u)*(-k)

# #             i = 1
# #             v = physics.q_ion[i]*u/Consts.k/physics.temperature
# #             a1 = (2.0*physics.q_ion[i]**2*physics.C_ion[i]/Consts.k
# #                   /physics.temperature*np.cosh(v)) #wrapped

# #             print('a')
# #             print(a)
# #             print(a1)
#             return a

#         return build_a

#     @staticmethod
#     def _build_f(physics):
#         #global build_f
#         def build_f(x,y,u): #generalized PB equation
#             n_ion = len(physics.c_ion)
#             f = np.zeros_like(x)
#             for i in range(n_ion):
#                 k = -physics.q_ion[i]/Consts.k/physics.temperature
#                 f += physics.q_ion[i]*physics.C_ion[i]*np.exp(k*u)*(1-k*u)

# #             i = 1
# #             v = physics.q_ion[i]*u/Consts.k/physics.temperature
# #             f1 = (-2.0*physics.q_ion[i]*physics.C_ion[i]
# #                   *(np.sinh(v)-np.cosh(v))*v) #wrapped

# #             print('f')
# #             print(f)
# #             print(f1)
#             return f

#         return build_f

    def _set_static_air(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.c_x['is_in_air'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_in_air'] = [[0.0]*n_rep for i in range(n_rep)]

        self.c_x['is_in_air'][-2][-2] = Consts.p
        self.c_y['is_in_air'][-2][-2] = Consts.p

    def _set_static_water(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.c_x['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.a['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.f['is_in_water'] = [0.0]*n_rep

        self.c_x['is_in_water'][-2][-2] = physics.perm_a
        self.c_y['is_in_water'][-2][-2] = physics.perm_a
        #self.a['is_in_water'][-2][-2] = StaticPDE._build_a(physics)
        #self.f['is_in_water'][-2] = StaticPDE._build_f(physics)
        self.a['is_in_water'][-2][-2] = 'func_a'
        self.f['is_in_water'][-2] = 'func_f'

    def _set_static_solid(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.c_x['is_in_solid'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_in_solid'] = [[0.0]*n_rep for i in range(n_rep)]
        
        self.c_x['is_in_solid'][-2][-2] = physics.perm_i
        self.c_y['is_in_solid'][-2][-2] = physics.perm_i

    def _set_static_stern_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.a['is_with_stern'] = [[0.0]*n_rep for i in range(n_rep)]
        self.f['is_with_stern'] = [0.0]*n_rep
        self.a['is_with_stern'][-1][-1] = 1.0
        self.f['is_with_stern'][-1] = 1.0*physics.sigma_solid

    def _set_static_mixed_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.q_s['is_with_mixed_bound'] = [[0.0]*n_rep for i in range(n_rep)]
        self.q_s['is_with_mixed_bound'][-2][-1] = -1.0

    def _set_static_inner_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.s_n['is_on_inner_bound'] = [None]*n_rep
        self.s_n['is_on_inner_bound'][-2] = physics.s_0

    def _set_static_metal_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        if physics.is_solid_metal:
            self.s_n['is_on_metal_bound'] = [None]*n_rep
            self.s_n['is_on_metal_bound'][-2] = 0.0
            
    def _set_static_outer_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.s_n['is_on_outer_bound'] = [0.0]*n_rep

    def _set_static_unused_nodes(self,physics):
        #to disable a zone 1) set PDE coefficients 0; 2) impose 0 Dirichlet B.C.
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.s_n['is_on_outside_domain'] = [None]*n_rep
        for i in range(n_rep):
            self.s_n['is_on_outside_domain'][i] = 0.0
        
        self.s_n['is_on_inside_domain'] = [None]*n_rep
        for i in range(n_ion):
            self.s_n['is_on_inside_domain'][i] = 0.0
        
        self.s_n['is_on_outside_stern'] = [None]*n_rep
        self.s_n['is_on_outside_stern'][-1] = 0.0


class PerturbPDE(PDE):
    def __init__(self,physics): #avoid long list of inputs
        attributes = ['c_x','c_y','alpha_x','alpha_y','a','s_n','g_s','q_s']
        for attr in attributes:
            self.__dict__[attr] = {}

        self.func_c = functools.partial(build_c,physics)
        self.func_alpha = functools.partial(build_alpha,physics)
        self.func_s = functools.partial(build_s,physics)

        self._set_perturb_air(physics)
        self._set_perturb_water(physics)
        self._set_perturb_solid(physics)
        self._set_perturb_stern_bound(physics)
        self._set_perturb_mixed_bound(physics)
        self._set_perturb_inner_bound(physics)
        self._set_perturb_metal_bound(physics)
        self._set_perturb_outer_bound(physics)
        self._set_perturb_unused_nodes(physics)
        self._set_dtype()
        self._set_shape()

#     @staticmethod
#     def _build_c(physics,i):
#         #global build_c
#         def build_c(x,y,pot):
#             v = np.exp(-physics.q_ion[i]*pot/Consts.k/physics.temperature)
#             c = physics.mu_a[i]*physics.z_ion[i]*physics.c_ion[i]*v #c or C_ion
#             return c

#         return build_c

#     @staticmethod
#     def _build_alpha(physics,i):
#         #global build_alpha
#         def build_alpha(x,y,grad):
#             alpha = physics.mu_a[i]*physics.z_ion[i]*grad #grad_x or grad_y
#             return alpha

#         return build_alpha

#     @staticmethod
#     def _build_s(physics):
#         #global build_s
#         def build_s(x,y,u):
#             s = -physics.e_0[0]*x-physics.e_0[1]*y
#             return s

#         return build_s

    def _set_perturb_air(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2

        self.c_x['is_in_air'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_in_air'] = [[0.0]*n_rep for i in range(n_rep)]

        self.c_x['is_in_air'][-2][-2] = Consts.p
        self.c_y['is_in_air'][-2][-2] = Consts.p
    
    def _set_perturb_water(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.c_x['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.alpha_x['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.alpha_y['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
        self.a['is_in_water'] = [[0.0]*n_rep for i in range(n_rep)]
#         func_c = functools.partial(build_c,physics)
#         func_alpha = functools.partial(build_alpha,physics)
        
        for i in range(n_ion):
            self.c_x['is_in_water'][i][i] = physics.Dm_a[i]
            self.c_y['is_in_water'][i][i] = physics.Dm_a[i]

            #self.c_x['is_in_water'][i][-2] = PerturbPDE._build_c(physics,i)
            #self.c_y['is_in_water'][i][-2] = PerturbPDE._build_c(physics,i)
            self.c_x['is_in_water'][i][-2] = 'func_c'
            self.c_y['is_in_water'][i][-2] = 'func_c'
            
            #self.alpha_x['is_in_water'][i][i] = PerturbPDE._build_alpha(physics,i)
            #self.alpha_y['is_in_water'][i][i] = PerturbPDE._build_alpha(physics,i)
            self.alpha_x['is_in_water'][i][i] = 'func_alpha'
            self.alpha_y['is_in_water'][i][i] = 'func_alpha'

            self.a['is_in_water'][i][i] = 1.0 #normalized frequency
            self.a['is_in_water'][-2][i] = -physics.z_ion[i]*Consts.f

        self.c_x['is_in_water'][-2][-2] = physics.perm_a
        self.c_y['is_in_water'][-2][-2] = physics.perm_a
    
    def _set_perturb_solid(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.c_x['is_in_solid'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_in_solid'] = [[0.0]*n_rep for i in range(n_rep)]
        
        self.c_x['is_in_solid'][-2][-2] = physics.perm_i
        self.c_y['is_in_solid'][-2][-2] = physics.perm_i

    def _set_perturb_stern_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.c_x['is_with_stern'] = [[0.0]*n_rep for i in range(n_rep)]
        self.c_y['is_with_stern'] = [[0.0]*n_rep for i in range(n_rep)]
        self.a['is_with_stern'] = [[0.0]*n_rep for i in range(n_rep)]
        
        self.c_x['is_with_stern'][-1][-2] = -physics.mu_s*physics.sigma_solid #normalized sigma_stern
        self.c_y['is_with_stern'][-1][-2] = -physics.mu_s*physics.sigma_solid #normalized sigma_stern
        
        self.c_x['is_with_stern'][-1][-1] = physics.Dm_s
        self.c_y['is_with_stern'][-1][-1] = physics.Dm_s
        
        self.a['is_with_stern'][-1][-1] = 1.0 #normalized frequency
    
    def _set_perturb_mixed_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.q_s['is_with_mixed_bound'] = [[0.0]*n_rep for i in range(n_rep)]
        self.q_s['is_with_mixed_bound'][-2][-1] = -1.0

    def _set_perturb_inner_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.s_n['is_on_inner_bound'] = [None]*n_rep
        self.s_n['is_on_inner_bound'][-2] = physics.s_0

    def _set_perturb_metal_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        if physics.is_solid_metal:
            self.s_n['is_on_metal_bound'] = [None]*n_rep
            self.s_n['is_on_metal_bound'][-2] = 0.0
            self.s_n['is_on_metal_bound'][-1] = 0.0
    
    def _set_perturb_outer_bound(self,physics):
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.s_n['is_on_outer_bound'] = [0.0]*n_rep
        #self.s_n['is_on_outer_bound'][-2] = PerturbPDE._build_s(physics)
        self.s_n['is_on_outer_bound'][-2] = 'func_s'
    
    def _set_perturb_unused_nodes(self,physics):
        #to disable a zone 1) set PDE coefficients 0; 2) impose 0 Dirichlet B.C.
        n_ion = len(physics.c_ion)
        n_rep = n_ion+2
        
        self.s_n['is_on_outside_domain'] = [None]*n_rep
        for i in range(n_rep):
            self.s_n['is_on_outside_domain'][i] = 0.0
        
        self.s_n['is_on_outside_water'] = [None]*n_rep
        for i in range(n_ion):
            self.s_n['is_on_outside_water'][i] = 0.0
        
        self.s_n['is_on_outside_stern'] = [None]*n_rep
        self.s_n['is_on_outside_stern'][-1] = 0.0


class StaticPB(StaticPDE): #Poisson-Boltzmann equation
    def __init__(self,physics): #avoid long list of inputs
        attributes = ['c_x','c_y','a','f','a_n','f_n','s_n','g_s']
        for attr in attributes:
            self.__dict__[attr] = {}

        self.func_a = functools.partial(build_a,physics)
        self.func_f = functools.partial(build_f,physics)

        self.c_x['is_in_water'] = [[physics.perm_a]]
        self.c_y['is_in_water'] = [[physics.perm_a]]
        self.a['is_on_water'] = [['func_a']] #can be either a or a_n 
        self.f['is_on_water'] = ['func_f'] #can be either f or f_n
        #self.a_n['is_on_water'] = [[StaticPDE._build_a(physics)]]
        #self.f_n['is_on_water'] = [StaticPDE._build_f(physics)]
        #self.a_n['is_on_water'] = [['func_a']]
        #self.f_n['is_on_water'] = ['func_f']
        self.g_s['is_with_mixed_bound'] = [1.0*physics.sigma_solid]
        self.s_n['is_on_outside_water'] = [0.0]
        self._set_dtype()
        self._set_shape()

    def decompose(self): #to use build_a and build_f externally
        pb1 = PDE()
        pb2 = PDE()

        pb1.c_x = copy.deepcopy(self.c_x)
        pb1.c_y = copy.deepcopy(self.c_y)
        pb1.g_s = copy.deepcopy(self.g_s)
        pb1.s_n = copy.deepcopy(self.s_n)
        
        pb2.c_x = {'is_in_water':[[0.0]]}
        pb2.c_y = {'is_in_water':[[0.0]]}
        pb2.a_n = {'is_on_water':[[1.0]]} #should not matter if a or a_n
        pb2.f_n = {'is_on_water':[1.0]} #should not matter if f or f_n
        
        pb1._set_dtype()
        pb1._set_shape()

        pb2._set_dtype()
        pb2._set_shape()
        
        return pb1,pb2


class PerturbPNP(PerturbPDE): #Poisson-Nernst-Planck equation
    def __init__(self,physics): #avoid long list of inputs
        super().__init__(physics)

    def decompose(self): #to use build_c and build_alpha externally
        n_rep = len(self.c_x[list(self.c_x.keys())[0]])
        
        pnp1 = PDE()
        pnp2 = PDE()
        
        pnp1.c_x = copy.deepcopy(self.c_x)
        pnp1.c_y = copy.deepcopy(self.c_y)
        pnp1.a = copy.deepcopy(self.a)
        pnp1.g_s = copy.deepcopy(self.g_s)
        pnp1.q_s = copy.deepcopy(self.q_s)

        pnp2.c_x = {'is_in_water':[[0.0]*n_rep for i in range(n_rep)]}
        pnp2.c_y = {'is_in_water':[[0.0]*n_rep for i in range(n_rep)]}
        
        pnp2.alpha_x = {'is_in_water':[[0.0]*n_rep for i in range(n_rep)]}
        pnp2.alpha_y = {'is_in_water':[[0.0]*n_rep for i in range(n_rep)]}

        for i in range(n_rep-2):
            pnp1.c_x['is_in_water'][i][-2] = 0.0
            pnp1.c_y['is_in_water'][i][-2] = 0.0

            pnp2.c_x['is_in_water'][i][-2] = 1.0 #for sparsity
            pnp2.c_y['is_in_water'][i][-2] = 1.0 #for sparsity

            pnp2.alpha_x['is_in_water'][i][i] = 1.0 #for sparsity
            pnp2.alpha_y['is_in_water'][i][i] = 1.0 #for sparsity

        pnp1._set_dtype()
        pnp1._set_shape()

        pnp2._set_dtype()
        pnp2._set_shape()

        return pnp1,pnp2

class Physics(Consts):
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)

        #self.C_ion = [val*Consts.n for val in self.c_ion]
        #self.Q_ion = [val*Consts.e for val in self.z_ion]
        self.Dm_a = [val*Consts.k*self.temperature/Consts.e
                       for val in self.mu_a] #wrapped
        self.Dm_s = self.mu_s*Consts.k*self.temperature/Consts.e
        self.perm_a = self.rel_perm_a*Consts.p
        self.perm_i = self.rel_perm_i*Consts.p

        #compute Debyle length for each ion species
        #compute thermal energy for each ion species
        #suggest initial surface charge density for solving PB equation
        #sigma_init is dominated by Debye length
        #if Debye length << sphere radius; vice versa        
        n_ion = len(self.c_ion)
        self.lambda_d = [0.0]*n_ion #Debye length
        self.Q = [0.0]*n_ion #thermal energy
        unsigned_sigma_init = [0.0]*n_ion
        for i in range(n_ion):
            self.Q[i] = Consts.k*self.temperature/abs(self.z_ion[i]*Consts.e)
            if self.c_ion[i]>0:
                self.lambda_d[i] = np.sqrt(self.perm_a*Consts.k*self.temperature
                                    /2/(self.z_ion[i]*Consts.e)**2/self.c_ion[i]
                                    /Consts.n) #wrapped
            else:
                self.lambda_d[i] = 0.0

            if self.radius_a>0:
                kappa = 1/self.lambda_d[i]+1/self.radius_a
            else:
                kappa = 1/self.lambda_d[i]

            unsigned_sigma_init[i] = 0.01*self.Q[i]*self.perm_a*kappa

        self.sigma_init = min(unsigned_sigma_init)*np.sign(self.sigma_solid)
#         for i in range(n_ion):
#             zeta = 0.01*Consts.k*self.temperature/abs(self.q_ion[i])
#             if self.radius_a>0:
#                 dl = 1/(1/self.lambda_d[i]+1/self.radius_a)
#             else:
#                 dl = self.lambda_d[i]
#             unsigned_sigma_init[i] = zeta*dl/self.perm_a

    @staticmethod
    def compute_k1(k,r):
        #modified spherical Bessel function of the second kind
        k1 = np.pi/2*np.exp(-k*r)*(1/(k*r)+1/(k*r)**2)
        return k1

    @staticmethod
    def grad_k1(k,r):
        #partial derivative of the modified spherical Bessel
        #function of the second kind evaluated at r=a
        k1 = Physics.compute_k1(k,r)
        dk1dr = -k*k1-np.pi/2*np.exp(-k*r)/r*(1/(k*r)+2/(k*r)**2)
        return dk1dr

    def compute_anpot(self,zeta,is_sigma,x,y,z):
        radius_a = self.radius_a
        lambda_d = self.lambda_d[0]

        #if radius_a is 0 return ansol_slab else return ansol_sphere
        if lambda_d==0:
            pass
        elif radius_a==0:
            pass
        else:
            if is_sigma:
                zeta = zeta/(1/lambda_d+1/radius_a)/self.perm_a
            dist = np.sqrt(x**2+y**2+z**2)
            pot = np.zeros((len(dist),4),dtype=float)
            mask = dist>radius_a
            pot[mask,0] = (zeta*radius_a*np.exp((radius_a-dist[mask])/lambda_d)
                           /dist[mask]) #wrapped
            pot[mask,1] = (-pot[mask,0]*(1/lambda_d+1/dist[mask])*x[mask]
                           /dist[mask]) #wrapped
            pot[mask,2] = (-pot[mask,0]*(1/lambda_d+1/dist[mask])*y[mask]
                           /dist[mask]) #wrapped
            pot[mask,3] = (-pot[mask,0]*(1/lambda_d+1/dist[mask])*z[mask]
                           /dist[mask]) #wrapped
            pot[~mask,0] = zeta

        print('SOLID PARTICLE RADIUS IS:',self.radius_a,'[m]')
        print('RELATIVE PERMITTIVITY OF ELECTROLYTE IS:',self.rel_perm_a,'[SI]')
        print('TEMPERATURE IS:',self.temperature,' [K]')
        print('ION COCENTRATION[0] AT INFINITY IS:',self.c_ion[0],'[mol/m^3]')
        print('ION VALENCE[0] IS:',self.z_ion[0])
        print('DEBYE LENGTH[0] IS:',self.lambda_d[0],'[m]')
        print('THERMAL ENERGY[0] IS:',self.Q[0],'[J]')
        print('POTENTIAL AT SOLID-LIQUID INTERFACE IS:',zeta, '[V]')
        print('')
        return pot

    def compute_anpnp(self,ratio,freq,x,y,alpha=0,beta=0,*args):
        n_ion = len(self.c_ion)
        if self.is_solid_metal and n_ion==2:
            dist = np.sqrt(x**2+y**2)
            conc = np.zeros((len(dist),n_ion),dtype=complex)
            pot = np.zeros(len(dist),dtype=complex)
            sigma = np.zeros(len(dist),dtype=complex)

            e_0 = np.sqrt(self.e_0[0]**2+self.e_0[1]**2)
            radius_a = self.radius_a
            perm_a = self.perm_a
            c_ion = np.r_[self.c_ion,0.0]
            mobility = self.mu_a[0]
            diffusion = self.Dm_a[0]
            lambda_d = self.lambda_d[0]
            lambda_1=np.sqrt(1j*freq*(2*np.pi)/diffusion+1/lambda_d**2)
            lambda_2=np.sqrt(1j*freq*(2*np.pi)/diffusion)

            a_1 = lambda_1*radius_a
            a_2 = lambda_2*radius_a
            f_2 = (a_1**2+2*a_1+2)/(a_1+1)
            f_3 = (a_2+1)/(a_2**2+2*a_2+2)
            f_1 = f_2*1j*freq*(2*np.pi)/diffusion*lambda_d**2

            numerator = 3*(1+beta*radius_a/diffusion*f_3) \
                +3*c_ion[2]/(c_ion[2]-2*c_ion[0])*(alpha/mobility-1) #wrapped
            denominator = c_ion[2]/(c_ion[2]-2*c_ion[0]) \
                *(f_1+alpha/mobility*(f_2-2)
                  +beta*radius_a*lambda_1**2/diffusion*lambda_d**2+2) \
                -(2+f_1)*(1+beta*radius_a/diffusion*f_3) #wrapped

            k_e = e_0*(1+numerator/denominator)
            f_e = f_2+(e_0+2*k_e)/(e_0-k_e)
            k_a = -(e_0*radius_a-k_e*radius_a)*lambda_1**2*perm_a \
                /2/Consts.f/Physics.compute_k1(lambda_1,radius_a) #wrapped
            k_b = (e_0-k_e)/Physics.grad_k1(lambda_2,radius_a) \
                *(lambda_1**2*perm_a/2/Consts.f*f_2
                  -mobility/diffusion*c_ion[0]*f_e) #wrapped
            k_m = -c_ion[2]/c_ion[0]*(e_0-k_e)\
                /Physics.grad_k1(lambda_2,radius_a) \
                /(1+beta*radius_a/diffusion*f_3) \
                *(lambda_1**2*perm_a/2/Consts.f*(f_2+beta*radius_a/diffusion)
                  -c_ion[0]/diffusion*(mobility-alpha)*f_e) #wrapped

            mask = dist>=radius_a
            rho = dist[mask]
            cosb = x[mask]/dist[mask]
            k1_1 = Physics.compute_k1(lambda_1,rho)
            k1_2 = Physics.compute_k1(lambda_2,rho)
            conc[mask,0] = -(k_a*k1_1+k_b*k1_2)*cosb
            conc[mask,1] = (c_ion[1]/c_ion[0]*k_a*k1_1+(k_b-k_m)*k1_2)*cosb
            #conc[mask,2] = (c_ion[2]/c_ion[0]*k_a*k1_1+k_m*k1_2)*cosb
            pot[mask] = (-2*Consts.f/lambda_1**2/perm_a*k_a*k1_1-e_0*rho
                         +k_e*radius_a**3/rho**2)*cosb #wrapped
        else:
            pass

        ansol = np.c_[conc,pot,sigma]
        return ansol


class Survey():
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)

