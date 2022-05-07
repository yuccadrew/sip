import numpy as np
import subprocess
import time
import matplotlib
from cycler import cycler

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

class Flags():
    pass

Flags.solid = 1
Flags.water = 2
Flags.air = 3
Flags.sw_interface = 1
Flags.aw_interface = 2
Flags.equipotential_surf = 3
Flags.axis_symmetry = 4
Flags.top_bound = 11
Flags.bottom_bound = 12
Flags.left_bound = 13
Flags.right_bound = 14

def build_polyfile(mesh_prefix,cpts,segs,holes,zones,dist_factor):
    #build the poly file
    f1 = open(mesh_prefix+'.poly','w')
    s1 = str(len(cpts))
    s2 = '  2 0 1  #verticies #dimensions  #attributes #boundary marker\n'
    f1.write(s1+s2)

    #write the vertices
    cnt = 1
    for i in range(len(cpts)):
        x = cpts[i,0]*dist_factor
        y = cpts[i,1]*dist_factor
        flag = cpts[i,2]
        s = "{0:6.0F} {1:20.6E} {2:20.6E} {3:6.0F}\n"
        f1.write(s.format(cnt,x,y,flag))
        cnt = cnt+1
    f1.write('\n')

    #write the segments
    cnt = 1
    s1 = str(len(segs))
    s2 = ' 1 #segments, boundary marker\n'
    f1.write(s1+s2)

    for i in range(len(segs)):
        ind_a = segs[i,0]+1
        ind_b = segs[i,1]+1
        flag = segs[i,2]
        s = "{0:6.0F} {1:6.0F} {2:6.0F} {3:6.0F}\n"
        f1.write(s.format(cnt,ind_a,ind_b,flag))
        cnt = cnt+1
    f1.write('\n')

    #write the holes
    f1.write('{0:6.0F}\n'.format(len(holes)))

    for i in range(len(holes)):
        x = holes[i,0]*dist_factor
        y = holes[i,1]*dist_factor
        s = '{0:6.0F} {1:20.6E} {2:20.6E} 1\n'
        f1.write(s.format(i+1,x,y))
    f1.write('\n')

    #write the area constraints for zones
    f1.write('{0:6.0F}\n'.format(len(zones)))

    for i in range(len(zones)):
        x = zones[i,0]*dist_factor
        y = zones[i,1]*dist_factor
        area = zones[i,2]*dist_factor**2
        s = '{0:6.0F} {1:20.6E} {2:20.6E} {3:6.0F} {4:12.6E}\n'
        f1.write(s.format(i+1,x,y,i+1,area))

    f1.write('\n')
    f1.write('# triangle -pnq30Aae '+mesh_prefix+'.poly \n')
    f1.close()

    return


def call_triangle(mesh_prefix,triangle_path):
    command = [triangle_path+' -pnq30Aae '+mesh_prefix+'.poly']
    process = subprocess.Popen(command,shell=True)
    process.wait()

    return

def import_nodes(mesh_prefix):
    print('Reading {}.1.node'.format(mesh_prefix))

    f = open('{}.1.node'.format(mesh_prefix))
    n_node = int(f.readline().split()[0])
    nodes = np.zeros((n_node,2),dtype=float)
    node_flags = np.zeros(n_node,dtype=int)
    for i in range(n_node):
        line = f.readline()
        nodes[i,:] = line.split()[1:3]
        node_flags[i] = line.split()[-1]
    f.close()

    return nodes,node_flags


def import_elements(mesh_prefix):
    print('Reading {}.1.ele'.format(mesh_prefix))

    f = open('{}.1.ele'.format(mesh_prefix))
    n_elem = int(f.readline().split()[0])
    elements = np.zeros((n_elem,3),dtype=int)
    elem_flags = np.zeros(n_elem,dtype=int)
    for i in range(n_elem):
        line = f.readline()
        elements[i,:] = line.split()[1:4]
        elem_flags[i] = line.split()[-1]
    f.close()

    #adjust indicies to start from zero
    elements = elements-1
    return elements,elem_flags


def import_edges(mesh_prefix):
    print('Reading {}.1.edge'.format(mesh_prefix))

    f = open('{}.1.edge'.format(mesh_prefix))
    n_edge = int(f.readline().split()[0])
    edges = np.zeros((n_edge,2),dtype=int)
    edge_flags = np.zeros(n_edge,dtype=int)
    for i in range(n_edge):
        line = f.readline()
        edges[i,:] = line.split()[1:3]
        edge_flags[i] = line.split()[-1]
    f.close()

    #adjust indicies to start from zero
    edges = edges-1
    return edges,edge_flags


def find_edges(nodes,x,y,flag):
    edges = np.zeros((len(x)-1,2),dtype=int)
    edge_flags = np.zeros(len(x)-1,dtype=int)
    for i in range(len(x)-1):
        ind_a = np.argmin((nodes[:,0]-x[i])**2+(nodes[:,1]-y[i])**2)
        ind_b = np.argmin((nodes[:,0]-x[i+1])**2+(nodes[:,1]-y[i+1])**2)
        diff_a = (nodes[ind_a,0]-x[i])**2+(nodes[ind_a,1]-y[i])**2
        diff_b = (nodes[ind_b,0]-x[i+1])**2+(nodes[ind_b,1]-y[i+1])**2
        if (diff_a+diff_b)>0:
            print('WARNING: ',edge_name, ' with error [',diff_a,diff_b,']')
        edges[i,0] = ind_a
        edges[i,1] = ind_b
        edge_flags[i] = flag

    return edges,edge_flags


def build_basis2d(nodes,elements):
    #compute shape function basis for triangular elements
    #input: nodes.shape (n_node,2)
    #input: elements.shape (n_nelem,3)
    #output: basis.shape(n_elem,3,3)
    #output: area.shape (n_elem,)
    
    print('Computing shape functions of triangular elements')
    print('This will take a minute')
    start = time.time()
    
    n_elem = len(elements)
    basis = np.zeros((n_elem,3,3),dtype=float)
    area = np.zeros(n_elem,dtype=float)
    
    x_node = nodes[elements,0] #(n_elem,3)
    y_node = nodes[elements,1] #(n_elem,3)
    
    for i in range(n_elem):
        A = np.ones((3,3),dtype=float)
        A[1,:] = x_node[i,:]
        A[2,:] = y_node[i,:]
        basis[i,:,:] = np.linalg.inv(A)
        area[i] = np.abs(np.linalg.det(A)/2.0)

    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return basis,area


def build_basis1d(nodes,edges):
    #compute shape function basis for line segments
    #need to convert 2D line to 1D line by
    #rotating 2D line along vector n to x
    
    #input: nodes.shape (n_node,2)
    #input: edges.shape (n_edge,2)
    #output: basis.shape(n_edge,2,2)
    #output: length.shape (n_edge,)
    
    print('Computing shape functions of line segments')
    print('This will take a minute')
    start = time.time()
    
    n_edge = len(edges)
    basis = np.zeros((n_edge,2,2),dtype=float)
    length = np.zeros(n_edge,dtype=float)
    
    x_node = nodes[edges,0] #(n_edge,2)
    y_node = nodes[edges,1] #(n_edge,2)
    
    for i in range(n_edge):
        #first compute rotation matrix R
        A = np.zeros((3,3),dtype=float)
        R = np.zeros((3,3),dtype=float)

        n = np.r_[x_node[i,1]-x_node[i,0],y_node[i,1]-y_node[i,0],0.0]
        norm_n = np.linalg.norm(n)
        n = n/norm_n
        x = np.r_[1.0,0.0,0.0]
        k = np.cross(n,x)
        norm_k = np.linalg.norm(k)
        if norm_k>1e-8: #consider improving threshold
            k = k/norm_k

            cos_b = np.dot(n,x)
            sin_b = np.sqrt(1-cos_b**2)

            A[0,:] = [0,-k[2],k[1]]
            A[1,:] = [k[2],0,-k[0]]
            A[2,:] = [-k[1],k[0],0]
            R = np.eye(3)+A.dot(sin_b)+A.dot(A.dot(1-cos_b))
        else:
            R = np.eye(3)

        #next compute basis and length
        B = np.ones((2,2),dtype=float)
        x_loc = np.zeros(2,dtype=float)
        x_loc[0] = R[0,0]*x_node[i,0]+R[0,1]*y_node[i,0]
        x_loc[1] = R[0,0]*x_node[i,1]+R[0,1]*y_node[i,1]
        B[1,:] = x_loc
        basis[i,:,:] = np.linalg.inv(B)
        length[i] = np.abs(x_loc[1]-x_loc[0])

    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return basis,length


class Mesh():
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)
    
    @classmethod
    def builder(cls,**kwargs): #avoid long list of inputs
        mesh = cls(**kwargs)
        mesh.dist_factor = 1.0
        build_polyfile(mesh.prefix,mesh.cpts,mesh.segs,mesh.holes,mesh.zones,
                       mesh.dist_factor)
        call_triangle(mesh.prefix,mesh.triangle)
        mesh.nodes,mesh.node_flags = import_nodes(mesh.prefix)
        mesh.elements,mesh.elem_flags = import_elements(mesh.prefix)
        mesh.edges,mesh.edge_flags = import_edges(mesh.prefix)
        print('')
        mesh._set_inds()
        mesh._set_basis()
        mesh._set_mids()
        mesh._set_rot_factor()
        
        return mesh
    
    @classmethod
    def importer(cls,**kwargs): #avoid long list of inputs
        mesh = cls(**kwargs)
        mesh.dist_factor = 1.0
        mesh.nodes,mesh.node_flags = import_nodes(mesh.prefix)
        mesh.elements,mesh.elem_flags = import_elements(mesh.prefix)
        mesh.edges,mesh.edge_flags = import_edges(mesh.prefix)
        mesh.nodes = mesh.nodes/1e6 #will be removed!!!
        print('')
        mesh._set_inds()
        mesh._set_basis()
        mesh._set_mids()
        mesh._set_rot_factor()

        return mesh
        
    def grad2d(self,f_n):
        print('Computing fields and gradients in elements')
        start = time.time()

        n_elem = len(self.elem_mids)
        f_out = np.zeros((n_elem,3),dtype=f_n.dtype)

        f_in = f_n[self.elements] #(n_elem,3)
        x = self.elem_mids[:,0] #(n_elem,)
        y = self.elem_mids[:,1] #(n_elem,)
        x_r = np.c_[x,x,x] #(n_elem,3)
        y_r = np.c_[y,y,y] #(n_elem,3)
        f_out[:,0] = np.sum((self.elem_basis[:,:,0]+self.elem_basis[:,:,1]*x_r
                             +self.elem_basis[:,:,2]*y_r)*f_in,axis=1) #wrapped
        f_out[:,1] = np.sum(f_in*self.elem_basis[:,:,1],axis=1)
        f_out[:,2] = np.sum(f_in*self.elem_basis[:,:,2],axis=1)

        elapsed=time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')

        return f_out

    def plot(self,f): #plots of colored lines
        x = self.nodes[self.edges,0]
        y = self.nodes[self.edges,1]
        vmin = min(0,min(f))
        vmax = max(f)
        cmap = matplotlib.cm.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        
        mask = abs(f)>0
        custom_cycler = cycler(color=cmap(norm(f[mask])))
        fig,ax = plt.subplots(figsize=(10,8))
        ax.set_prop_cycle(custom_cycler)
        ax.plot(x[mask,:].T,y[mask,:].T)
        pc = ax.scatter(x[mask,0],y[mask,0],s=2,c=f[mask],
                        vmin=vmin,vmax=vmax,cmap='viridis') #wrapped
        fig.colorbar(pc,ax=ax,location='right')
        ax.set_aspect('equal')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ax.set_xlim(min(xmin,ymin),max(xmax,ymax))
        ax.set_ylim(min(xmin,ymin),max(xmax,ymax))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Zero values are shaded')
    
    def scatter(self,f_n): #scatter plots of colored points
        if len(f_n)==len(self.elements):
            x = self.elem_mids[:,0]
            y = self.elem_mids[:,1]
        elif len(f_n)==len(self.edges):
            x = self.edge_mids[:,0]
            y = self.edge_mids[:,1]
        else:
            x = self.nodes[:,0]
            y = self.nodes[:,1]

        fig,ax = plt.subplots(figsize=(10,8))
        sc = ax.scatter(x,y,s=200,c=f_n,cmap='coolwarm')
        fig.colorbar(sc,ax=ax,location='right')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

    def tripcolor(self,f_in,vmin=[],vmax=[]): #tripcolor plots of colored elements
        if len(f_in)==len(self.nodes):
            f = self.grad2d(f_in)[:,0]            
        else:
            f = f_in
        
        if vmin==[]:
            vmin = min(f)
        
        if vmax==[]:
            vmax = max(f)

        x = self.nodes[:,0]
        y = self.nodes[:,1]        
        fig,ax=plt.subplots(figsize=(10,8))
        tpc=ax.tripcolor(x,y,self.elements,facecolors=f,edgecolor='none',
                         vmin=vmin,vmax=vmax,cmap='jet') #wrapped
        fig.colorbar(tpc,ax=ax,location='right')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.show()
    
    def visualize(self,elem_flags=[],edge_flags=[],xlim=[],ylim=[]):
        print('THE NUMBER OF NODES IS: {0:6.0F}'.format(len(self.nodes)))
        print('THE NUMBER OF ELEMENTS IS: {0:6.0F}'.format(len(self.elements)))
        print('THE NUMBER OF EDGES IS: {0:6.0F}'.format(len(self.edges)))
        print('')
        print('node_flags',np.unique(self.node_flags))
        print('elem_flags',np.unique(self.elem_flags))
        print('edge_flags',np.unique(self.edge_flags))
        print('')

        #plot all triangles in the background
        fig,ax = plt.subplots(figsize=(10,8))
        x = self.nodes[:,0]
        y = self.nodes[:,1]
        ax.triplot(x,y,self.elements[:,:],linewidth=0.2,
                   color='tab:blue',alpha=0.5) #wrapped
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        #plot specified elements
        for i in range(len(elem_flags)):
            mask = self.elem_flags==elem_flags[i]
            if np.sum(mask)>0:
                ax.triplot(x,y,self.elements[mask,:],linewidth=0.2,
                           color='tab:blue',alpha=1.0) #wrapped

        #plot all edges in the background
        x = self.nodes[self.edges,0]
        y = self.nodes[self.edges,1]
        for key in Flags.__dict__.keys():
            if not '__' in key:
                mask = self.edge_flags==Flags.__dict__[key]
                ax.plot(x[mask,:].T,y[mask,:].T,color='tab:orange',alpha=0.2)
        
        #plot specified edges
        for i in range(len(edge_flags)):
            mask = self.edge_flags==edge_flags[i]
            ax.plot(x[mask,:].T,y[mask,:].T,color='tab:orange',alpha=1.0)

        if len(xlim)==2:
            ax.set_xlim(xlim)

        if len(ylim)==2:
            ax.set_ylim(ylim)

        plt.show()

    def _set_inds(self):
        #define basic mesh indexing attributes
        self.is_in_air = self.elem_flags==Flags.air
        self.is_in_water = self.elem_flags==Flags.water
        self.is_in_solid = self.elem_flags==Flags.solid
        self.is_inside_domain = (self.is_in_solid|self.is_in_water)|self.is_in_air

        self.is_with_stern = self.edge_flags==Flags.sw_interface
        self.is_with_equipotential = self.edge_flags==Flags.equipotential_surf
        self.is_with_axis_symmetry = self.edge_flags==Flags.axis_symmetry
        
        self.is_with_top_bound = self.edge_flags==Flags.top_bound
        self.is_with_bottom_bound = self.edge_flags==Flags.bottom_bound
        self.is_with_left_bound = self.edge_flags==Flags.left_bound
        self.is_with_right_bound = self.edge_flags==Flags.right_bound
        
        #define advanced mesh indexing attributes (default False)
        self.is_on_air = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_water = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_solid = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_inside_domain = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_stern = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_equipotential = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_top_bound = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_bottom_bound = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_left_bound = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_right_bound = np.zeros(len(self.nodes),dtype=bool)
        self.is_on_outer_bound = np.zeros(len(self.nodes),dtype=bool)

        #define advanced mesh indexing attributes (default True)
        self.is_on_outside_domain = np.ones(len(self.nodes),dtype=bool)
        self.is_on_outside_water = np.ones(len(self.nodes),dtype=bool)
        self.is_on_outside_stern = np.ones(len(self.nodes),dtype=bool)
        
        #define alias
        self.is_with_mixed_bound = self.is_with_stern
        self.is_on_metal_bound = self.is_on_stern
        self.is_on_inner_bound = self.is_on_equipotential

        #compute advanced mesh indexing attributes (change to True)
        mask = self.is_in_air
        ind_n = np.unique(self.elements[mask,:].flatten(order='C'))
        self.is_on_air[ind_n] = True
        
        mask = self.is_in_water
        ind_n = np.unique(self.elements[mask,:].flatten(order='C'))
        self.is_on_water[ind_n] = True

        mask = self.is_in_solid
        ind_n = np.unique(self.elements[mask,:].flatten(order='C'))
        self.is_on_solid[ind_n] = True
        
        mask = self.is_inside_domain
        ind_n = np.unique(self.elements[mask,:].flatten(order='C'))
        self.is_on_inside_domain[ind_n] = True

        mask = self.is_with_stern
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_stern[ind_n] = True
        
        mask = self.is_with_equipotential
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_equipotential[ind_n] = True
        
        mask = self.is_with_top_bound
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_top_bound[ind_n] = True

        mask = self.is_with_bottom_bound
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_bottom_bound[ind_n] = True
        
        mask = self.is_with_left_bound
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_left_bound[ind_n] = True

        mask = self.is_with_right_bound
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_right_bound[ind_n] = True

        mask = self.is_with_top_bound
        mask = mask|self.is_with_bottom_bound
        mask = mask|self.is_with_left_bound
        mask = mask|self.is_with_right_bound
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_outer_bound[ind_n] = True
        
        #compute advanced mesh indexing attributes (change to False)
        mask = self.is_inside_domain
        ind_n = np.unique(self.elements[mask,:].flatten(order='C'))
        self.is_on_outside_domain[ind_n] = False
        
        mask = self.is_in_water
        ind_n = np.unique(self.elements[mask,:].flatten(order='C'))
        self.is_on_outside_water[ind_n] = False
        
        mask = self.is_with_stern
        ind_n = np.unique(self.edges[mask,:].flatten(order='C'))
        self.is_on_outside_stern[ind_n] = False
    
    def _set_basis(self):  
        n_elem = len(self.elements)
        n_edge = len(self.edges)
        self.elem_basis = np.zeros((n_elem,3,3),dtype=float)
        self.elem_area = np.zeros(n_elem,dtype=float)
        self.edge_basis = np.zeros((n_edge,2,2),dtype=float)
        self.edge_len = np.zeros(n_edge,dtype=float)
        
        #compute shape functions for triangular elements
        mask = self.is_inside_domain
        basis,area = build_basis2d(self.nodes,self.elements[mask,:])
        self.elem_basis[mask,:,:] = basis
        self.elem_area[mask] = area

        #compute shape functions for line segments
        mask = self.is_with_stern
        basis,length = build_basis1d(self.nodes,self.edges[mask,:])
        self.edge_basis[mask,:,:] = basis
        self.edge_len[mask] = length
    
    def _set_mids(self):
        n_elem = len(self.elements)
        n_edge = len(self.edges)
        self.elem_mids = np.zeros((n_elem,2),dtype=float)
        self.edge_mids = np.zeros((n_edge,2),dtype=float)
        
        #compute middle points of triangular elements
        x = self.nodes[self.elements,0] #(n_elem,3)
        y = self.nodes[self.elements,1] #(n_elem,3)
        self.elem_mids[:,0] = np.sum(x,axis=1)/3.0
        self.elem_mids[:,1] = np.sum(y,axis=1)/3.0

        #compute middle points of line segments
        x = self.nodes[self.edges,0] #(n_edge,2)
        y = self.nodes[self.edges,1] #(n_edge,2)
        self.edge_mids[:,0] = np.sum(x,axis=1)/2.0
        self.edge_mids[:,1] = np.sum(y,axis=1)/2.0
    
    def _set_rot_factor(self):
        #determine the coefficient scaling factor based on the axis of symmetry
        if self.axis_symmetry=='X':
            self.node_factor = self.nodes[:,1]
            self.elem_factor = self.elem_mids[:,1]
            self.edge_factor = self.edge_mids[:,1] #needs to be verified
        elif self.axis_symmetry=='Y':
            self.node_factor = self.nodes[:,0]
            self.elem_factor = self.elem_mids[:,0]
            self.edge_factor = self.edge_mids[:,0] #needs to be verified
        else:
            self.node_factor = 1.0
            self.elem_factor = 1.0
            self.edge_factor = 1.0

class Complex():
    def __init__(self,**kwargs): #avoid long list of inputs
        for key,value in kwargs.items():
            setattr(self,key,value)
    
    @classmethod
    def init_slab(cls,**kwargs): #avoid long list of inputs
        slab = cls(**kwargs)

        #define size of the bounding box
        radius_b = max(slab.radius_air,slab.radius_water,slab.radius_solid)
        height_b = max(slab.height_air,slab.height_water,slab.height_solid)
        height_water = slab.height_water

        #define control points, line segments, holes, and zone constraints
        cpts = np.zeros((0,3),dtype=float) #x/y/z of control points
        segs = np.zeros((0,3),dtype=int) #ind_a/ind_b/flag of lines
        holes = np.zeros((0,2),dtype=float) #x/y of holes
        zones = np.zeros((0,3),dtype=float) #x/y/area of zone constraints

        #define holes and zone constraints
        if height_water<height_b:
            x = np.r_[0.0,0.0,0.0] #solid/water/air
            y = np.r_[-height_b,height_water,height_water*2+height_b]/2.0
            area = np.r_[1.0,1.0,1.0]*(radius_b/20.0)**2
            zones = np.r_[zones,np.c_[x,y,area]]
        else:
            x = np.r_[0.0,0.0] #solid/water
            y = np.r_[-height_b,height_water]/2.0
            area = np.r_[1.0,1.0]*(radius_b/20.0)**2
            zones = np.r_[zones,np.c_[x,y,area]]

        #define the outer boundary box
        x = np.r_[-radius_b,radius_b,radius_b,-radius_b]
        y = np.r_[-height_b,-height_b,height_b,height_b]
        cpts = np.r_[cpts,np.c_[x,y,np.ones_like(x)*1]]

        #define the solid-water interface
        x = np.r_[-radius_b,radius_b]
        y = np.r_[0.0,0.0]
        cpts = np.r_[cpts,np.c_[x,y,np.ones_like(x)*1]]

        #define the air-water interface
        if height_water<height_b:
            x = np.r_[-radius_b,radius_b]
            y = np.r_[height_water,height_water]
            cpts = np.r_[cpts,np.c_[x,y,np.ones_like(x)*1]]

        #define the segments on the bottom boundary
        x = np.r_[-radius_b,radius_b]
        y = np.r_[-height_b,-height_b]
        edges,edge_flags = find_edges(cpts,x,y,Flags.bottom_bound)
        segs = np.r_[segs,np.c_[edges,edge_flags]]

        #define the segments on the right boundary
        x = np.r_[radius_b,radius_b,radius_b]
        y = np.r_[-height_b,height_water,height_b]
        edges,edge_flags = find_edges(cpts,x,y,Flags.right_bound)
        segs = np.r_[segs,np.c_[edges,edge_flags]]

        #define the segments on the top boundary
        x = np.r_[-radius_b,radius_b]
        y = np.r_[height_b,height_b]
        edges,edge_flags = find_edges(cpts,x,y,Flags.top_bound)
        segs = np.r_[segs,np.c_[edges,edge_flags]]

        #define the segments on the left boundary
        x = np.r_[-radius_b,-radius_b,-radius_b]
        y = np.r_[-height_b,height_water,height_b]
        edges,edge_flags = find_edges(cpts,x,y,Flags.left_bound)
        segs = np.r_[segs,np.c_[edges,edge_flags]]

        #define the segments on the solid-water interface
        x = np.r_[-radius_b,radius_b]
        y = np.r_[0.0,0.0]
        edges,edge_flags = find_edges(cpts,x,y,Flags.sw_interface)
        segs = np.r_[segs,np.c_[edges,edge_flags]]

        #define the segments on the air-water interface
        if height_water<height_b:
            x = np.r_[-radius_b,radius_b]
            y = np.r_[height_water,height_water]
            edges,edge_flags = find_edges(cpts,x,y,Flags.aw_interface)
            segs = np.r_[segs,np.c_[edges,edge_flags]]
        
        slab.cpts = cpts
        slab.segs = segs
        slab.holes = holes
        slab.zones = zones
        
        return slab

    def visualize(self):
        fig,ax = plt.subplots(figsize=(5,4))
        x = self.cpts[self.segs[:,:-1],0]
        y = self.cpts[self.segs[:,:-1],1]
        ax.plot(x.T,y.T,'-',color='tab:blue')
        ax.plot(self.cpts[:,0],self.cpts[:,1],'.',color='tab:orange')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Zoom-out')
        plt.show()

