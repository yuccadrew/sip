import numpy as np
import time
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')


class StaticSolution():
    def __init__(self,mesh,domain,robin,dirichlet):
        #declare all attributes first
        K = [] #placeholder
        b = [] #placeholder
        sol = [] #placeholder

        #update materials
        domain.c_x[:] = 1.0*mesh.elem_factor
        domain.c_y[:] = 1.0*mesh.elem_factor

        dist = np.sqrt(mesh.nodes[:,0]**2+mesh.nodes[:,1]**2)
        f_n = np.zeros_like(dist)
        mask = dist>0
        f_n[mask] = np.pi/2*(1/dist[mask]*np.sin(np.pi*dist[mask]/2.0)
                             +np.pi/2*np.cos(np.pi*dist[mask]/2.0))
        f_n[~mask] = np.pi/2*(np.pi/2+np.pi/2)
        domain.f_n[:,0] = f_n*mesh.node_factor

        mask = mesh.is_on_outer_bound
        dirichlet.s_n[mask,0] = np.cos(np.pi*dist[mask]/2.0)

        #update K1 and K2 and b1 and b2
        domain.update(mesh)

        sigma_diffuse = 0.0
        K = domain.K1+domain.K2+robin.K1+robin.K2
        b = domain.b1+domain.b2+robin.b1*sigma_diffuse+robin.b2
        self.K,self.b = dirichlet.set_1st_kind_bc(K,b)
        
        print('Calling sparse linear system solver')
        start = time.time()
        self.K.eliminate_zeros()
        self.sol = spsolve(self.K,self.b)
        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')

#         self.K1,self.K2,self.b1,self.b2 = assemble_Ke2d(mesh,domain)
#         self.K3,self.K4,self.b3,self.b4 = assemble_Ks2d(mesh,robin)
#         self.with_1st_kind_bc = dirichlet.with_1st_kind_bc
#         self.s_n = dirichlet.s_n
    def spy(self):
        fig,ax=plt.subplots()
        ax.spy(self.K)
        plt.show()

    def visualize(self):
        pass


class Solution():
    def __init__(self,mesh,domain,robin,dirichlet):
        n_node = len(mesh.nodes)
        n_rep = domain.c_x.shape[1]
        
        #declare all attributes first
        K = [] #placeholder
        b = [] #placeholder
        sol = [] #placeholder

        domain.build_system(mesh)
        sigma_diffuse = 0.0
        K = domain.K1+domain.K2
        b = domain.b1+domain.b2

        self.K,self.b = dirichlet.set_first_kind_bc(K,b)

        print('Calling sparse linear system solver')
        start = time.time()
        self.sol = spsolve(self.K,self.b)
        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')

