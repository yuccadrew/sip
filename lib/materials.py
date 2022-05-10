import numpy as np
from scipy import constants
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

class Consts():
    pass

Consts.e = constants.e #elementary charge
Consts.p = constants.epsilon_0 #vacuum permittivity
Consts.a = constants.N_A #Avogadro constant
Consts.k = constants.k #Boltzmann constant
Consts.f = constants.value('Faraday constant') #Faraday constant


# Using lambda arguments: expression
flatten_list = lambda irregular_list:[element for item in irregular_list \
                                      for element in flatten_list(item)] \
                   if type(irregular_list) is list else [irregular_list]


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
                        if callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0)
                        else:
                            self.__dict__[attr][mask,i] = val[i]

                        continue
                    
                    for j in range(len(val[i])):
                        #print(attr,ky,'[',i,',',j,']',val[i][j])
                        if callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0)
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

        self.c_x = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.c_y = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.alpha_x = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.alpha_y = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.beta_x = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.beta_y = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.gamma_x = np.zeros((n_edge,n_rep),dtype=float)
        self.gamma_y = np.zeros((n_edge,n_rep),dtype=float)
        self.a = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        self.f = np.zeros((n_edge,n_rep),dtype=float)
        
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
                        if callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0)
                        else:
                            self.__dict__[attr][mask,i] = val[i]
                        
                        continue

                    for j in range(len(val[i])):
                        #print(attr,ky,'[',i,',',j,']',val[i][j])
                        if callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0)
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
        
        self.g_s = np.zeros((n_edge,n_rep),dtype=float)
        self.q_s = np.zeros((n_edge,n_rep,n_rep),dtype=float)
        
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
                        if callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0)
                        else:
                            self.__dict__[attr][mask,i] = val[i]
                        continue
                        
                    for j in range(len(val[i])):
                        #print(attr,ky,'[',i,',',j,']',val[i][j])
                        if callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0)
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
        self.s_n = np.zeros((n_node,n_rep),dtype=float)

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

                        if callable(val[i]):
                            self.__dict__[attr][mask,i] = val[i](x,y,0)
                        else:
                            self.__dict__[attr][mask,i] = val[i]
                        
                        continue

                    for j in range(len(val[i])):
                        if val[i][j] == None:
                            continue

                        #print(attr,ky,'[',i,',',j,']',val_i[j])
                        self.on_first_kind_bc[mask,i,j] = True

                        if callable(val[i][j]):
                            self.__dict__[attr][mask,i,j] = val[i][j](x,y,0)
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
        for key,value in kwargs.items():
            setattr(self,key,value)
    
    @classmethod
    def init_static(cls,physics): #avoid long list of inputs
        stat = cls()
        keys = ['c_x','c_y','a','f','s_n','q_s']
        for ky in keys:
            stat.__dict__[ky] = {}

        stat._set_static_air(physics)
        stat._set_static_water(physics)
        stat._set_static_solid(physics)
        stat._set_static_stern_bound(physics)
        stat._set_static_mixed_bound(physics)
        stat._set_static_inner_bound(physics)
        stat._set_static_metal_bound(physics)
        stat._set_static_outer_bound(physics)
        stat._set_static_unused_nodes(physics)
        return stat
    
    @classmethod
    def init_perturb(cls,physics):
        pass

    @staticmethod
    def _build_a(physics):
        def build_a(x,y,u): #only works for two types of ions
#             K_B = Consts.k
#             N_A = Consts.a
#             E_C = Consts.e
#             zval = 1.0
#             pot_0 = np.zeros_like(x)
#             cinf = physics.c_ion[0] #ion concentration
#             ze = zval*E_C #ion valence times elementary charge
#             u2d_scale = ze*pot_0[:]/K_B/physics.temperature #scaled potential in elements
#             a = 2*ze**2*N_A*cinf/K_B/physics.temperature*np.cosh(u2d_scale)            
#             f = -2*ze*N_A*cinf*(np.sinh(u2d_scale)-np.cosh(u2d_scale)*u2d_scale)
            
            for i in range(1,2):
                v = physics.Q_ion[i]*u/Consts.k/physics.temperature
                a = (2.0*physics.Q_ion[i]**2*physics.C_ion[i]/Consts.k
                     /physics.temperature*np.cosh(v)) #wrapped
            return a

        return build_a

    @staticmethod
    def _build_f(physics):
        def build_f(x,y,u): #only works for two types of ions
#             K_B = Consts.k
#             N_A = Consts.a
#             E_C = Consts.e
#             zval = 1.0
#             pot_0 = np.zeros_like(x)
#             cinf = physics.c_ion[0] #ion concentration
#             ze = zval*E_C #ion valence times elementary charge
#             u2d_scale = ze*pot_0[:]/K_B/physics.temperature #scaled potential in elements
#             a = 2*ze**2*N_A*cinf/K_B/physics.temperature*np.cosh(u2d_scale)            
#             f = -2*ze*N_A*cinf*(np.sinh(u2d_scale)-np.cosh(u2d_scale)*u2d_scale)

            #u2d_scale = ze*pot_0[:]/K_B/temperature
            #f = -2*ze*N_A*cinf*(np.sinh(u2d_scale)-np.cosh(u2d_scale)*u2d_scale)
            for i in range(1,2):
                v = physics.Q_ion[i]*u/Consts.k/physics.temperature
                f = (-2.0*physics.Q_ion[i]*physics.C_ion[i]
                     *(np.sinh(v)-np.cosh(v))*v) #wrapped
            return f

        return build_f

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
        self.a['is_in_water'][-2][-2] = PDE._build_a(physics)
        self.f['is_in_water'][-2] = PDE._build_f(physics)
    
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
        self.f['is_with_stern'][-1] = 1.0
    
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

    def visualize(self,*args):
        if args:
            attributes = args
        else:
            attributes = self.__dict__.keys()
        
        attr_1 = ['c_x','c_y','alpha_x','alpha_y','beta_x','beta_y',
                      'a','a_n','q_s'] #ndim equal 3
        attr_2 = ['gamma_x','gamma_y','f','f_d','f_n','g_s','s_n'] #ndim equal 2

        for attr in attributes:
            if attr not in attr_1+attr_2:
                continue

            for ky in self.__dict__[attr].keys():
                print('{}[\'{}\']: '.format(attr,ky))
                val = self.__dict__[attr][ky]

                for i in range(len(val)):
                    print('[',end='')
                    if type(val[i]) is not list:
                        if callable(val[i]):
                            print('{0:>8}()'.format(val[i].__name__),end='')
                        elif val[i] is None:
                            print('{0:>10}'.format('None'),end='')
                        else:
                            print('{0:10.2E}'.format(val[i]),end='')

                        print(']')
                        continue
                    
                    for j in range(len(val[i])):
                        #print(val[i][j])
                        if callable(val[i][j]):
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

        self._set_static_air(physics)
        self._set_static_water(physics)
        self._set_static_solid(physics)
        self._set_static_stern_bound(physics)
        self._set_static_mixed_bound(physics)
        self._set_static_inner_bound(physics)
        self._set_static_metal_bound(physics)
        self._set_static_outer_bound(physics)
        self._set_static_unused_nodes(physics)

    @staticmethod
    def _build_a(physics):
        def build_a(x,y,u): #only works for two types of ions
#             K_B = Consts.k
#             N_A = Consts.a
#             E_C = Consts.e
#             zval = 1.0
#             pot_0 = np.zeros_like(x)
#             cinf = physics.c_ion[0] #ion concentration
#             ze = zval*E_C #ion valence times elementary charge
#             u2d_scale = ze*pot_0[:]/K_B/physics.temperature #scaled potential in elements
#             a = 2*ze**2*N_A*cinf/K_B/physics.temperature*np.cosh(u2d_scale)            
#             f = -2*ze*N_A*cinf*(np.sinh(u2d_scale)-np.cosh(u2d_scale)*u2d_scale)
            
            for i in range(1,2):
                v = physics.Q_ion[i]*u/Consts.k/physics.temperature
                a = (2.0*physics.Q_ion[i]**2*physics.C_ion[i]/Consts.k
                     /physics.temperature*np.cosh(v)) #wrapped
            return a
        
        return build_a

    @staticmethod
    def _build_f(physics):
        def build_f(x,y,u): #only works for two types of ions
#             K_B = Consts.k
#             N_A = Consts.a
#             E_C = Consts.e
#             zval = 1.0
#             pot_0 = np.zeros_like(x)
#             cinf = physics.c_ion[0] #ion concentration
#             ze = zval*E_C #ion valence times elementary charge
#             u2d_scale = ze*pot_0[:]/K_B/physics.temperature #scaled potential in elements
#             a = 2*ze**2*N_A*cinf/K_B/physics.temperature*np.cosh(u2d_scale)            
#             f = -2*ze*N_A*cinf*(np.sinh(u2d_scale)-np.cosh(u2d_scale)*u2d_scale)

            #u2d_scale = ze*pot_0[:]/K_B/temperature
            #f = -2*ze*N_A*cinf*(np.sinh(u2d_scale)-np.cosh(u2d_scale)*u2d_scale)
            for i in range(1,2):
                v = physics.Q_ion[i]*u/Consts.k/physics.temperature
                f = (-2.0*physics.Q_ion[i]*physics.C_ion[i]
                     *(np.sinh(v)-np.cosh(v))*v) #wrapped
            return f

        return build_f

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
        self.a['is_in_water'][-2][-2] = PDE._build_a(physics)
        self.f['is_in_water'][-2] = PDE._build_f(physics)
    
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
        self.f['is_with_stern'][-1] = 1.0
    
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

        self._set_perturb_air(physics)
        self._set_perturb_water(physics)
        self._set_perturb_solid(physics)
        self._set_perturb_stern_bound(physics)
        self._set_perturb_mixed_bound(physics)
        self._set_perturb_inner_bound(physics)
        self._set_perturb_metal_bound(physics)
        self._set_perturb_outer_bound(physics)
        self._set_perturb_unused_nodes(physics)

    @staticmethod
    def _build_c(physics,i):
        def build_c(x,y,pot):
            v = np.exp(-physics.Q_ion[i]*pot/Consts.k/physics.temperature)
            c = physics.mu_a[i]*physics.z_ion[i]*physics.c_ion[i]*v #c or C_ion
            return c

        return build_c

    @staticmethod
    def _build_alpha(physics,i):
        def build_alpha(x,y,grad):
            alpha = physics.mu_a[i]*physics.z_ion[i]*grad #grad_x or grad_y
            return alpha

        return build_alpha

    @staticmethod
    def _build_s(physics):
        def build_s(x,y,u):
            s = -physics.e_0[0]*x-physics.e_0[1]*y
            return s

        return build_s

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
        
        for i in range(n_ion):
            self.c_x['is_in_water'][i][i] = physics.Diff_a[i]
            self.c_y['is_in_water'][i][i] = physics.Diff_a[i]

            self.c_x['is_in_water'][i][-2] = PerturbPDE._build_c(physics,i)
            self.c_y['is_in_water'][i][-2] = PerturbPDE._build_c(physics,i)
            
            self.alpha_x['is_in_water'][i][i] = PerturbPDE._build_alpha(physics,i)
            self.alpha_y['is_in_water'][i][i] = PerturbPDE._build_alpha(physics,i)
            
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
        
        self.c_x['is_with_stern'][-1][-2] = physics.mu_s #normalized sigma_stern
        self.c_y['is_with_stern'][-1][-2] = physics.mu_s #normalized sigma_stern
        
        self.c_x['is_with_stern'][-1][-1] = physics.Diff_s
        self.c_y['is_with_stern'][-1][-1] = physics.Diff_s
        
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
        self.s_n['is_on_outer_bound'][-2] = PerturbPDE._build_s(physics)
    
    def _set_perturb_unused_nodes(self,physics):
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


class Physics(Consts):
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)

        self.C_ion = [val*Consts.a for val in self.c_ion]
        self.Q_ion = [val*Consts.e for val in self.z_ion]
        self.Diff_a = [val*Consts.k*self.temperature/Consts.e
                       for val in self.mu_a] #wrapped
        self.Diff_s = self.mu_s*Consts.k*self.temperature/Consts.e
        self.perm_a = self.rel_perm_a*Consts.p
        self.perm_i = self.rel_perm_i*Consts.p

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

