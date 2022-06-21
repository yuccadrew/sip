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

#def build_polyfile(cpts,segs,holes,zones,mesh_prefix,dist_factor):
def build_polyfile(mesh_prefix,cpts,segs,holes,zones,dist_factor):
    print('Writing '+mesh_prefix+'.poly')
    print('')
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


def discretize_rho(lambda_d,rho_min,rho_max):
    #obtain fine grid next to the solid-liquid interface
    #use 16 points between 0.02*debye_length and 10*debye_length
    #reduce debye_length if water film is too thin
    #lambda_d = min(9e-9,height_water)
    #print('DEBYE LENGTH IS: %.2e nm'%(lambda_d*1e9))

    rho = np.logspace(np.log10(0.02),1,16)*lambda_d
    dr = np.diff(np.log10(rho/lambda_d))[0]
    #print('min(np.diff(rho) %.2e nm'%(min(np.diff(rho))*1e9))
    #print('height_water %.2e nm'%height_water)
    #print('')
    
    #use the logarithmic interval above to discretize the entire space
    #which extends to radius_solid for this slab example
    rho = np.power(10,np.arange(np.log10(0.02)-dr,1,dr))*lambda_d
    cnt = 0
    while rho[-1]<(rho_max-rho_min):
        cnt = cnt+1
        rho = np.power(10,np.arange(np.log10(0.02)-dr,1+dr*cnt,dr))*lambda_d

    #adjust the starting and ending points in the discretization
    rho = np.r_[rho_min,rho[:-2]+rho_min,rho_max]
    #no need to refine diffuse layer if there is no diffuse layer
    #rho = np.r_[rho_min,rho_max]
    return rho


class Mesh():
    def __init__(self,*args,**kwargs): #avoid long list of inputs
        if args:
            pass

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)
            self.dist_factor = 1.0
            self.nodes,self.node_flags = import_nodes(self.prefix)
            self.elements,self.elem_flags = import_elements(self.prefix)
            self.edges,self.edge_flags = import_edges(self.prefix)
            self.nodes = self.nodes*self.unscale_factor #will be removed!!!
            print('THE NUMBER OF NODES IS: %d'%len(self.nodes))
            print('THE NUMBER OF ELEMENTS IS: %d'%len(self.elements))
            print('THE NUMBER OF EDGES IS: %d'%len(self.edges))
            print('node_flags',np.unique(self.node_flags))
            print('elem_flags',np.unique(self.elem_flags))
            print('edge_flags',np.unique(self.edge_flags))
            print('')
            self._set_inds()
            self._set_basis()
            self._set_mids()
            self._set_rot_factor()
            #self._set_edge_neigh()

    def save(self,hf_group):
        for attr in self.__dict__.keys():
            if type(self.__dict__[attr]) is str:
                hf_group.create_dataset(attr,data=self.__dict__[attr]) #placeholder
            else:
                hf_group.create_dataset(attr,data=self.__dict__[attr])

    @classmethod
    def load(cls,hf_group):
        mesh = cls()
        for attr in hf_group.keys():
            setattr(mesh,attr,np.array(hf_group[attr]))
            if mesh.__dict__[attr].dtype=='object':
                mesh.__dict__[attr] = str(mesh.__dict__[attr].astype(str))

        return mesh

    @classmethod
    def builder(cls,*args,**kwargs): #avoid long list of inputs
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
    def importer(cls,*args,**kwargs): #avoid long list of inputs
        mesh = cls(**kwargs)
        mesh.dist_factor = 1.0
        mesh.nodes,mesh.node_flags = import_nodes(mesh.prefix)
        mesh.elements,mesh.elem_flags = import_elements(mesh.prefix)
        mesh.edges,mesh.edge_flags = import_edges(mesh.prefix)
        mesh.nodes = mesh.nodes*mesh.unscale_factor #will be removed!!!
        print('')
        mesh._set_inds()
        mesh._set_basis()
        mesh._set_mids()
        mesh._set_rot_factor()
        #mesh._set_edge_neigh()

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

    def flux2d(self,f_x,f_y):
        flux = np.zeros((len(self.edges),2),dtype=float)
        for i in range(2): #possible two adjacent elements
            nodes = self.nodes
            mask = self.edge_to_elem[:,i]>=0
            
            #vector u points from edges[:,0] to edges[:,1]
            edges = self.edges[mask,:]
            u_x = nodes[edges[:,1],0]-nodes[edges[:,0],0]
            u_y = nodes[edges[:,1],1]-nodes[edges[:,0],1]
            u_len = np.sqrt(u_x**2+u_y**2)
            u_x = u_x/u_len
            u_y = u_y/u_len
            
            #vector v points from elem_mids to edges[:,0]
            elements = self.elements[self.edge_to_elem[mask,i],:]
            elem_mids = self.elem_mids[self.edge_to_elem[mask,i],:]
            v_x = nodes[edges[:,0],0]-elem_mids[:,0]
            v_y = nodes[edges[:,0],1]-elem_mids[:,1]
            #v_len = np.sqrt(v_x**2+v_y**2)
            #v_x = v_x/v_len
            #v_y = v_y/v_len

            #vector n = v-dot(u,v)u
            n_x = v_x-(u_x*v_x+u_y*v_y)*u_x
            n_y = v_y-(u_x*v_x+u_y*v_y)*u_y
            n_len = np.sqrt(n_x**2+n_y**2)
            n_x = n_x/n_len
            n_y = n_y/n_len
            #print(np.r_[n_x,n_y])

            try:
                flux[mask,i] = (f_x[self.edge_to_elem[mask,i]]*n_x
                                +f_y[self.edge_to_elem[mask,i]]*n_y) #wrapped
            except:
                flux[mask,i] = f_x*n_x+f_y*n_y #for test purpose f_x=1,f_y=1

        return flux

    def to_spherical(self,is_nodal=True):
        if is_nodal:
            x = self.nodes[:,0]
            y = self.nodes[:,1]
            z = 0
        else:
            x = self.elem_mids[:,0]
            y = self.elem_mids[:,1]
            z = 0
        rho = np.sqrt(x**2+y**2+z**2) #radial distance
        #theta = np.arccos(z/rho) #polar angle
        #phi = np.arctan2(y/x)
        phi = np.zeros_like(rho)+np.pi/2 #azimuthal angle
        mask = x>0
        phi[mask] = np.arctan(y[mask]/x[mask])
        mask = x<0
        phi[mask] = np.arctan(y[mask]/x[mask])+np.pi
        #return rho,theta,phi
        return rho,phi

    def plot(self,f,cmap='viridis'): #plots of colored lines
        x = self.nodes[self.edges,0]
        y = self.nodes[self.edges,1]
        vmin = min(0,min(f))
        vmax = max(f)
        cmap = matplotlib.cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        
        mask = abs(f)>0
        custom_cycler = cycler(color=cmap(norm(f[mask])))
        fig,ax = plt.subplots(figsize=(10,8))
        ax.set_prop_cycle(custom_cycler)
        ax.plot(x[mask,:].T,y[mask,:].T)
        pc = ax.scatter(x[mask,0],y[mask,0],s=2,c=f[mask],
                        vmin=vmin,vmax=vmax,cmap=cmap) #wrapped
        fig.colorbar(pc,ax=ax,location='right')
        ax.set_aspect('equal')
        xmin,xmax = ax.get_xlim()
        ymin,ymax = ax.get_ylim()
        ax.set_xlim(min(xmin,ymin),max(xmax,ymax))
        ax.set_ylim(min(xmin,ymin),max(xmax,ymax))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Zero values are shaded')
    
    def scatter(self,f_n,xlim=[],ylim=[],cmap='coolwarm'): #scatter plots of colored points
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
        sc = ax.scatter(x,y,s=200,c=f_n,cmap=cmap)
        fig.colorbar(sc,ax=ax,location='right')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        if len(xlim)==2:
            ax.set_xlim(xlim)

        if len(ylim)==2:
            ax.set_ylim(ylim)

        plt.show()

    def tricontourf(self,f_in,xlim=[],ylim=[],vmin=[],vmax=[],title=[],
                    levels=20,cmap='turbo',logscale=False,xunit='m',yunit='m'): #wrapped #contourf plots
        if xunit=='nanometer' or xunit=='nm':
            xunit = 'nm'
            x_factor = 1e9
        elif xunit=='micrometer' or xunit=='um':
            xunit = '$\mu$m'
            x_factor = 1e6
        elif xunit=='millimeter' or xunit=='mm':
            xunit = 'mm'
            x_factor = 1e3
        else:
            xunit = 'm'
            x_factor = 1.0
        
        if yunit=='nanometer' or yunit=='nm':
            yunit = 'nm'
            y_factor = 1e9
        elif yunit=='micrometer' or yunit=='um':
            yunit = '$\mu$m'
            y_factor = 1e6
        elif yunit=='millimeter' or yunit=='mm':
            yunit = 'mm'
            y_factor = 1e3
        else:
            yunit = 'm'
            y_factor = 1.0

        if len(f_in)==len(self.nodes):
            n_ind = np.where(~self.is_on_outside_domain)[0]
            x = self.nodes[n_ind,0]
            y = self.nodes[n_ind,1]
            f = f_in[n_ind]
        else:
            x = self.elem_mids[:,0]
            y = self.elem_mids[:,1]
            f = f_in

        #!!!found bug when vmin and vmax are the same!!!
        if vmin==[]:
            vmin = min(f)

        if vmax==[]:
            vmax = max(f)

        if logscale:
            #z = numpy.ma.masked_invalid(z)
            #vmin, vmax = z.min(), z.max()
            #z = z.filled(fill_value=-999)
            vmin = max(vmin,1e-12)
            f = np.ma.masked_where(f<=0,f)
            f = f.filled(fill_value=1e-12)
            level_exp = np.arange(np.floor(np.log10(vmin)),
                                  np.ceil(np.log10(vmax)+1))
            levels = np.ceil(np.log10(levels)).astype(int)
            level_exp = np.linspace(level_exp[0],level_exp[-1],
                                    (len(level_exp)-1)*levels+1)
            levels = np.power(10,level_exp)
            norm = matplotlib.colors.LogNorm()
            parser = {'levels':levels,'cmap':cmap,'norm':norm,'extend':'min'}
        else:
            levels = np.linspace(vmin,vmax,levels)
            norm = None
            parser = {'levels':levels,'cmap':cmap,'norm':norm,
                      'vmin':vmin,'vmax':vmax}

        x = x*x_factor
        y = y*y_factor
        fig,ax = plt.subplots()
        cs = ax.tricontourf(x,y,f,**parser)
        cbar = fig.colorbar(cs,ax=ax)
        ax.set_aspect('equal')
        ax.set_xlabel('X ('+xunit+')')
        ax.set_ylabel('Y ('+yunit+')')

        if self.axis_symmetry=='X':
            ax.tricontourf(x,-y,f,**parser)
        elif self.axis_symmetry=='Y':
            ax.tricontourf(-x,y,f,**parser)

        if len(xlim)==2:
            xlim[0] = xlim[0]*x_factor
            xlim[1] = xlim[1]*x_factor
            ax.set_xlim(xlim)

        if len(ylim)==2:
            ylim[0] = ylim[0]*x_factor
            ylim[1] = ylim[1]*x_factor
            ax.set_ylim(ylim)

        if len(title)>0:
            ax.set_title(title)
        plt.show()

    def tripcolor(self,f_in,xlim=[],ylim=[],vmin=[],vmax=[],title=[],
                  edgecolor='none',cmap='YlGnBu_r',logscale=False,
                  contour=False,xunit='m',yunit='m'): #wrapped #tripcolor plots of colored elements
        if xunit=='nanometer' or xunit=='nm':
            xunit = 'nm'
            x_factor = 1e9
        elif xunit=='micrometer' or xunit=='um':
            xunit = '$\mu$m'
            x_factor = 1e6
        elif xunit=='millimeter' or xunit=='mm':
            xunit = 'mm'
            x_factor = 1e3
        else:
            xunit = 'm'
            x_factor = 1.0
        
        if yunit=='nanometer' or yunit=='nm':
            yunit = 'nm'
            y_factor = 1e9
        elif yunit=='micrometer' or yunit=='um':
            yunit = '$\mu$m'
            y_factor = 1e6
        elif yunit=='millimeter' or yunit=='mm':
            yunit = 'mm'
            y_factor = 1e3
        else:
            yunit = 'm'
            y_factor = 1.0

        if len(f_in)==len(self.nodes):
            x_in = self.nodes[:,0]
            y_in = self.nodes[:,1]
            grads = self.grad2d(f_in)
            f = grads[:,0]
        else:
            x_in = self.elem_mids[:,0]
            y_in = self.elem_mids[:,1]
            f = f_in

        #!!!found bug when vmin and vmax are the same!!!
        if vmin==[]:
            vmin = min(f)

        if vmax==[]:
            vmax = max(f)
        
        if logscale:
            norm = matplotlib.colors.LogNorm(vmin,vmax)
            parser = {'edgecolor':edgecolor,'cmap':cmap,'norm':norm}
        else:
            norm = None
            parser = {'edgecolor':edgecolor,'cmap':cmap,'norm':norm,
                      'vmin':vmin,'vmax':vmax}

        x = self.nodes[:,0]*x_factor
        y = self.nodes[:,1]*y_factor
        fig,ax=plt.subplots(figsize=(10,8))
        tpc = ax.tripcolor(x,y,self.elements,facecolors=f,**parser)
        fig.colorbar(tpc,ax=ax,location='right')
        ax.set_aspect('equal')
        ax.set_xlabel('X ('+xunit+')')
        ax.set_ylabel('Y ('+yunit+')')

        if self.axis_symmetry=='X':
            ax.tripcolor(x,-y,self.elements,facecolors=f,**parser)
        elif self.axis_symmetry=='Y':
            ax.tripcolor(-x,y,self.elements,facecolors=f,**parser)

        if contour:
            x_in = x_in*x_factor
            y_in = y_in*y_factor
            ax.tricontour(x_in,y_in,f_in,colors='white')
            
            if self.axis_symmetry=='X':
                ax.tricontour(x_in,-y_in,f_in,colors='white')
            
            if self.axis_symmetry=='Y':
                ax.tricontour(-x_in,y_in,f_in,colors='white')

        if len(xlim)==2:
            xlim[0] = xlim[0]*x_factor
            xlim[1] = xlim[1]*x_factor
            ax.set_xlim(xlim)

        if len(ylim)==2:
            ylim[0] = ylim[0]*x_factor
            ylim[1] = ylim[1]*x_factor
            ax.set_ylim(ylim)

        if len(title)>0:
            ax.set_title(title)
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
        
        if self.axis_symmetry=='X':
            ax.triplot(x,-y,self.elements[:,:],linewidth=0.2,
                       color='tab:blue',alpha=0.25) #wrapped
        
        if self.axis_symmetry=='Y':
            ax.triplot(-x,y,self.elements[:,:],linewidth=0.2,
                       color='tab:blue',alpha=0.25) #wrapped

        #plot specified elements
        for i in range(len(elem_flags)):
            mask = self.elem_flags==elem_flags[i]
            if np.sum(mask)>0:
                ax.triplot(x,y,self.elements[mask,:],linewidth=0.2,
                           color='tab:blue',alpha=1.0) #wrapped

        #plot all edges in the background
        x = self.nodes[self.edges,0]
        y = self.nodes[self.edges,1]
        for attr in Flags.__dict__.keys():
            if not '__' in attr:
                mask = self.edge_flags==Flags.__dict__[attr]
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
        self.is_on_axis_symmetry = np.zeros(len(self.nodes),dtype=bool)
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
        ind_n = np.unique(self.elements[mask,:].ravel())
        self.is_on_air[ind_n] = True
        
        mask = self.is_in_water
        ind_n = np.unique(self.elements[mask,:].ravel())
        self.is_on_water[ind_n] = True

        mask = self.is_in_solid
        ind_n = np.unique(self.elements[mask,:].ravel())
        self.is_on_solid[ind_n] = True
        
        mask = self.is_inside_domain
        ind_n = np.unique(self.elements[mask,:].ravel())
        self.is_on_inside_domain[ind_n] = True

        mask = self.is_with_stern
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_stern[ind_n] = True
        
        mask = self.is_with_equipotential
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_equipotential[ind_n] = True
        
        mask = self.is_with_axis_symmetry
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_axis_symmetry[ind_n] = True
        
        mask = self.is_with_top_bound
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_top_bound[ind_n] = True

        mask = self.is_with_bottom_bound
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_bottom_bound[ind_n] = True
        
        mask = self.is_with_left_bound
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_left_bound[ind_n] = True

        mask = self.is_with_right_bound
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_right_bound[ind_n] = True

        mask = self.is_with_top_bound
        mask = mask|self.is_with_bottom_bound
        mask = mask|self.is_with_left_bound
        mask = mask|self.is_with_right_bound
        ind_n = np.unique(self.edges[mask,:].ravel())
        self.is_on_outer_bound[ind_n] = True
        
        #compute advanced mesh indexing attributes (change to False)
        mask = self.is_inside_domain
        ind_n = np.unique(self.elements[mask,:].ravel())
        self.is_on_outside_domain[ind_n] = False
        
        mask = self.is_in_water
        ind_n = np.unique(self.elements[mask,:].ravel())
        self.is_on_outside_water[ind_n] = False
        
        mask = self.is_with_stern
        ind_n = np.unique(self.edges[mask,:].ravel())
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

    def _set_edge_neigh(self): #efficienty to be improved
        print('Finding adjacent elements for line segments')
        print('This will take a minute')
        start = time.time()
        n_elem = len(self.elements)
        n_edge = len(self.edges)
        self.edge_to_elem = np.zeros((n_edge,2),dtype=int)-1
        self.elem_to_edge = [[]*1 for i in range(n_elem)]

        for i in range(n_edge):
            mask_1 = self.edges[i,0]==self.elements[:,0]
            mask_1 = mask_1|(self.edges[i,0]==self.elements[:,1])
            mask_1 = mask_1|(self.edges[i,0]==self.elements[:,2])
            
            mask_2 = self.edges[i,1]==self.elements[:,0]
            mask_2 = mask_2|(self.edges[i,1]==self.elements[:,1])
            mask_2 = mask_2|(self.edges[i,1]==self.elements[:,2])
            
            ind = np.where(mask_1&mask_2)[0]
            if len(ind)==1:
                self.edge_to_elem[i,0] = ind[0]
                self.elem_to_edge[ind[0]].append(i)
            elif len(ind)==2:
                self.edge_to_elem[i,0] = ind[0]
                self.edge_to_elem[i,1] = ind[1]
                self.elem_to_edge[ind[0]].append(i)
                self.elem_to_edge[ind[1]].append(i)
            #else:
            #    print(i,ind)
        
        self.elem_to_edge = np.array(self.elem_to_edge,dtype=int)
        #sort of self.edge_to_elem by columns
        #left column: flux2d positive if (f_x,f_y) is (1,1)
        #right column: flux2d negative if (f_x,f_y) is (1,1)
        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')

    def _set_elem_neigh(self):#slower than set_edge_neigh
        print('Finding edges for triangular elements')
        print('This will take a minute')
        start = time.time()
        n_elem = len(self.elements)
        n_edge = len(self.edges)
        self.elem_to_edge = np.zeros((n_elem,3),dtype=int)
        self.edge_to_elem = [[]*1 for i in range(n_edge)]

        for i in range(n_elem):
            ind_a = [0,1,2]
            ind_b = [1,2,0]
            for j in range(3):
                mask_1 = self.elements[i,ind_a[j]]==self.edges[:,0]
                mask_1 = mask_1|(self.elements[i,ind_a[j]]==self.edges[:,1])

                mask_2 = self.elements[i,ind_b[j]]==self.edges[:,0]
                mask_2 = mask_2|(self.elements[i,ind_b[j]]==self.edges[:,1])

                ind = np.where(mask_1&mask_2)[0]
                self.elem_to_edge[i,j] = ind[0]
                self.edge_to_elem[ind[0]].append(i)

        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')


class Complex():
    def __init__(self,*args,**kwargs): #avoid long list of inputs
        if args:
            pass

        if kwargs:
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


class Sphere():
    def __init__(self,*args,**kwargs):
        if args:
            pass

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)

    def build(self):
        pass

    def visualize(self):
        pass


class Slab():
    def __init__(self,*args,**kwargs):
        if args:
            pass

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)

    def build(self):
        pass

    def visualize(self):
        pass


class Probe():
    def __init__(self,*args,**kwargs):
        if args:
            pass

        if kwargs:
            for key,value in kwargs.items():
                setattr(self,key,value)
            self.build()

    def build(self):
        #user inputs of background geometry
        radius_air = self.radius_air #radius of the air
        height_air = self.height_air #height of the air
        radius_water = self.radius_air #radius of thin water film
        height_water = self.height_water #thickness of thin water film
        radius_solid = self.radius_air #radius of the solid
        height_solid = self.height_air #height of the solid

        #user inputs of probe geometry
        radius_tip = self.radius_tip #radius of probe tip
        offset_tip = self.offset_tip #offset between probe tip and sw interface
        radius_cone = self.radius_cone #radius of probe cone
        height_cone = self.height_cone #height of probe cone
        radius_disk = self.radius_disk #radius of probe disk
        height_disk = self.height_disk #height of probe disk

        #discretize rho
        lambda_d = min(9e-9,height_water)
        rho = discretize_rho(lambda_d,rho_min=0,rho_max=radius_solid)

        #insert air-water interface into the discretization
        mask = rho<height_water
        rho = np.r_[rho[mask],height_water,rho[~mask]]
        #ind = np.argmin((rho-height_water)**2)
        #rho[ind] = height_water

        #print out discretization
        #print('See radial discretization below')
        #print(rho)

        #generate mesh
        #This script only works for the same height and radius for air and solid
        #X is the axis of symmetry
        #Y is the longitudinal axis
        cpts = np.zeros((0,3)) #coord_x/coord_y/flag of control points
        segs = np.zeros((0,3)) #ind_a/ind_b/flag of line segmenets
        holes = np.zeros((0,2)) #coord_x/coord_y
        zones = np.zeros((0,3)) #coord_x/coord_y/area

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #define the lowermost, rightmost, and topmost boundary points
        radius_b = max(radius_air,radius_solid)
        height_b = max(height_air,height_solid)
        x = np.r_[0,radius_b,radius_b,0]
        y = np.r_[-height_b,-height_b,height_b,height_b]
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*1]] #node flag of 1

        #define the leftmost boundary points (on the axis of symmetry)
        #skip points at two ends
        mask = (rho>0)&(rho<offset_tip)
        y = np.r_[-np.flipud(rho[:-1]),rho[mask],offset_tip,
                  offset_tip+2*radius_tip,
                  offset_tip+2*radius_tip+height_cone,
                  offset_tip+2*radius_tip+height_cone+height_disk]
        x = np.zeros_like(y)
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*1]] #node flag of 1

        #-----------------------------------------------------------------------
        #define the top edge points of the solid
        #skip edge points on the axis of symmetry
        x = np.r_[rho[1:-1],radius_solid]
        y = np.r_[0*rho[1:-1],0]
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the top edge points of the water
        #skip edge points on the axis of symmetry
        x = np.r_[rho[1:-1],radius_water]
        y = np.ones_like(x)*height_water
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #define the bottom edge points of the water
        #skip the bottom edge points on the axis of symmetry
        x = np.r_[radius_water]
        y = np.r_[0]
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the edge points on the tip surface
        #skip edge points on the axis of symmetry
        nA = 32
        ns = nA+1-2
        dA = np.pi/nA
        phi = np.arange(1,ns+1)*dA-np.pi/2 #half the circle
        x = radius_tip*np.cos(phi)+0.0
        y = radius_tip*np.sin(phi)+offset_tip+radius_tip
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the edge points on the cone surface
        x = np.r_[radius_cone]
        y = np.r_[height_cone]+offset_tip+2*radius_tip
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define inner control points along the cantilever
        x = np.r_[radius_disk,radius_disk]
        y = np.r_[0,height_disk]+offset_tip+2*radius_tip+height_cone
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #define the segments on the lowermost boundary
        x = np.r_[0,radius_b]
        y = np.r_[-radius_b,-radius_b]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.bottom_bound]]

        #define the segments on the rightmost boundary
        x = np.r_[radius_b,radius_b]
        y = np.r_[-radius_b,radius_b]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.right_bound]]

        #define the segments on the topmost boundary
        x = np.r_[0,radius_b]
        y = np.r_[radius_b,radius_b]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.top_bound]]

        #define the segments on the leftmost boundary (axis of symmetry)
        mask = (rho>0)&(rho<offset_tip)
        y = np.r_[-radius_b,-np.flipud(rho),rho[mask],offset_tip,
                  offset_tip+2*radius_tip,offset_tip+2*radius_tip+height_cone,
                  offset_tip+2*radius_tip+height_cone+height_disk,radius_b]
        x = np.zeros_like(y)
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.axis_symmetry]]

        #-----------------------------------------------------------------------
        #define the segments on the top edge of the solid (solid-liquid interface)
        x = np.r_[rho[:-1],radius_solid]
        y = np.zeros_like(x)
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.sw_interface]]

        #-----------------------------------------------------------------------
        #define the segments on the right edge of the water
        if radius_water<radius_air:
            x = np.r_[radius_water,radius_water]
            y = np.r_[0,height_water]
            for i in range(len(x)-1):
                ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
                ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
                segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.aw_interface]]

        #define the segments on the top edge of the water
        #x = np.r_[0,radius_water]
        #y = np.r_[height_water,height_water]
        x = np.r_[rho[rho<radius_water],radius_water]
        y = np.zeros_like(x)+height_water
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.aw_interface]]

        #-----------------------------------------------------------------------
        #define the segments along the bottom tip surface
        nA = 32
        ns = nA+1-2
        dA = np.pi/nA
        phi = np.arange(0,nA//2+1)*dA-np.pi/2 #half the circle
        #print(phi*180/np.pi)
        x = radius_tip*np.cos(phi)+0.0
        y = radius_tip*np.sin(phi)+offset_tip+radius_tip
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

    #     #define the segments along the top tip surface
    #     nA = 32
    #     ns = nA+1-2
    #     dA = np.pi/nA
    #     phi = np.arange(nA//2,nA+1)*dA-np.pi/2 #half the circle
    #     #print(phi*180/np.pi)
    #     x = radius_tip*np.cos(phi)+0.0
    #     y = radius_tip*np.sin(phi)+offset_tip+radius_tip
    #     for i in range(len(x)-1):
    #         ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
    #         ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
    #         segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

        #-----------------------------------------------------------------------
        #define the right segments along the cone surface
        x = np.r_[radius_tip,radius_cone]
        y = np.r_[0,height_cone+radius_tip]+offset_tip+radius_tip
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

    #     #define the top segments along the cone surface
    #     x = np.r_[0,radius_cone]
    #     y = np.r_[height_cone,height_cone]+offset_tip+2*radius_tip
    #     for i in range(len(x)-1):
    #         ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
    #         ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
    #         segs = np.r_[segs,np.c_[ind_a,ind_b,100]]

        #-----------------------------------------------------------------------
        #define the segments along the remaining cantilever surface
        x = np.r_[0,radius_disk,radius_disk,radius_cone]
        y = np.r_[height_disk,height_disk,0,0]+offset_tip+2*radius_tip+height_cone
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #define markers for holes and zones
    #     x = np.r_[0,0,0,0,0,0]+radius_tip/2 #solid,water,tip,cone,arm,air
    #     y = np.r_[-height_solid/2,height_water/2,offset_tip+radius_tip,
    #               offset_tip+2*radius_tip+height_cone/2,
    #               offset_tip+2*radius_tip+height_cone+height_disk/2,
    #               height_air/4]
    #     area = np.r_[1,1,1,100,100,900]
    #     zones = np.r_[zones,np.c_[x,y,area]]
        x = np.r_[0]+radius_tip/2 #cantilever tip
        y = np.r_[offset_tip+radius_tip]
        holes = np.r_[holes,np.c_[x,y]]

        x = np.r_[0,0,0]+radius_tip/2 #solid,water,air
        y = np.r_[-height_solid/2,height_water/2,
                  offset_tip+2*radius_tip+height_cone+height_disk*2]
        area = np.r_[100,1,100]*1e-12
        zones = np.r_[zones,np.c_[x,y,area]]
        self.cpts = cpts
        self.segs = segs
        self.holes = holes
        self.zones = zones

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #write poly file and call triangle
        mesh_prefix = self.mesh_prefix
        dist_factor = self.dist_factor

        if self.build_mesh:
            build_polyfile(mesh_prefix,cpts,segs,holes,zones,dist_factor)
            call_triangle(mesh_prefix,'triangle')

    def visualize(self):
        nodes,node_flags = import_nodes(self.mesh_prefix)
        elements,elem_flags = import_elements(self.mesh_prefix)
        edges,edge_flags = import_edges(self.mesh_prefix)
        nodes = nodes/self.dist_factor #unscale nodes
        print('THE NUMBER OF NODES IS: %d'%len(nodes))
        print('THE NUMBER OF ELEMENTS IS: %d'%len(elements))
        print('THE NUMBER OF EDGES IS: %d'%len(edges))
        print('node_flags',np.unique(node_flags))
        print('elem_flags',np.unique(elem_flags))
        print('edge_flags',np.unique(edge_flags))
        print('')

        disp_factor = 1e6
        cpts = self.cpts
        segs = self.segs
        x = cpts[segs[:,:-1].astype(int),0]
        y = cpts[segs[:,:-1].astype(int),1]

        radius_air = self.radius_air
        radius_tip = self.radius_tip
        radius_disk = self.radius_disk
        height_water = self.height_water

        fig,ax = plt.subplots(2,2,figsize=(8,8))
        axs = ax.flatten()

        axs[0].plot(x.T*disp_factor,y.T*disp_factor,'-',color='tab:blue')
        axs[0].plot(cpts[:,0]*disp_factor,cpts[:,1]*disp_factor,'.',color='tab:orange')
        axs[0].set_xlabel('X ($\mu$m)')
        axs[0].set_ylabel('Y ($\mu$m)')
        axs[0].set_aspect('equal')
        axs[0].set_xlim(-radius_air*1.1*disp_factor,radius_air*1.1*disp_factor)
        axs[0].set_ylim(-radius_air*1.1*disp_factor,radius_air*1.1*disp_factor)
        axs[0].set_title('Zoom Out')

        axs[1].plot(x.T*disp_factor,y.T*disp_factor,'-',color='tab:blue')
        axs[1].plot(cpts[:,0]*disp_factor,cpts[:,1]*disp_factor,'.',color='tab:orange')
        axs[1].set_xlabel('X ($\mu$m)')
        axs[1].set_ylabel('Y ($\mu$m)')
        axs[1].set_aspect('equal')
        axs[1].set_xlim(-radius_disk*1.1*disp_factor,radius_disk*1.1*disp_factor)
        axs[1].set_ylim(-radius_disk*1.1*disp_factor,radius_disk*1.1*disp_factor)
        axs[1].set_title('Zoom In: Cantilever')

        axs[2].plot(x.T*disp_factor,y.T*disp_factor,'-',color='tab:blue')
        axs[2].plot(cpts[:,0]*disp_factor,cpts[:,1]*disp_factor,'.',color='tab:orange')
        axs[2].set_xlabel('X ($\mu$m)')
        axs[2].set_ylabel('Y ($\mu$m)')
        axs[2].set_aspect('equal')
        axs[2].set_xlim(-radius_tip*4*disp_factor,radius_tip*4*disp_factor)
        axs[2].set_ylim(-radius_tip*4*disp_factor,radius_tip*4*disp_factor)
        axs[2].set_title('Zoom In: Tip')

        axs[3].plot(x.T*disp_factor,y.T*disp_factor,'-',color='tab:blue')
        axs[3].plot(cpts[:,0]*disp_factor,cpts[:,1]*disp_factor,'.',color='tab:orange')
        axs[3].set_xlabel('X ($\mu$m)')
        axs[3].set_ylabel('Y ($\mu$m)')
        axs[3].set_aspect('equal')
        axs[3].set_xlim(-height_water*10*disp_factor,height_water*10*disp_factor)
        axs[3].set_ylim(-height_water*10*disp_factor,height_water*10*disp_factor)
        axs[3].set_title('Zoom In: Water')

        plt.tight_layout()
        plt.show()

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        x = nodes[:,0]*disp_factor
        y = nodes[:,1]*disp_factor

        fig,ax=plt.subplots(2,2,figsize=(8,8),dpi=80)
        axs=ax.flatten()

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[0].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[0].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue',alpha=0.5)
        axs[0].set_xlabel('X ($\mu$m)')
        axs[0].set_ylabel('Y ($\mu$m)')
        axs[0].set_aspect('equal')
        axs[0].set_xlim(-radius_air*1.1*disp_factor,radius_air*1.1*disp_factor)
        axs[0].set_ylim(-radius_air*1.1*disp_factor,radius_air*1.1*disp_factor)
        axs[0].set_title('Zoom Out')

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[1].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[1].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue',alpha=0.5)
        mask=(elem_flags>=1)&(elem_flags<=5)
        axs[1].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:orange')
        axs[1].set_xlabel('X ($\mu$m)')
        axs[1].set_ylabel('Y ($\mu$m)')
        axs[1].set_aspect('equal')
        axs[1].set_xlim(-radius_disk*1.1*disp_factor,radius_disk*1.1*disp_factor)
        axs[1].set_ylim(-radius_disk*1.1*disp_factor,radius_disk*1.1*disp_factor)
        axs[1].set_title('Zoom In: Cantilever')

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[2].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[2].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue',alpha=0.5)
        mask=(elem_flags>=1)&(elem_flags<=5)
        axs[2].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:orange')
        axs[2].set_xlabel('X ($\mu$m)')
        axs[2].set_ylabel('Y ($\mu$m)')
        axs[2].set_aspect('equal')
        axs[2].set_xlim(-radius_tip*4*disp_factor,radius_tip*4*disp_factor)
        axs[2].set_ylim(-radius_tip*4*disp_factor,radius_tip*4*disp_factor)
        axs[2].set_title('Zoom In: Tip')

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[3].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[3].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue',alpha=0.5)
        mask=(elem_flags>=2)&(elem_flags<=2)
        axs[3].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:orange')
        axs[3].set_xlabel('X ($\mu$m)')
        axs[3].set_ylabel('Y ($\mu$m)')
        axs[3].set_aspect('equal')
        axs[3].set_xlim(-height_water*10*disp_factor,height_water*10*disp_factor)
        axs[3].set_ylim(-height_water*10*disp_factor,height_water*10*disp_factor)
        axs[3].set_title('Zoom In: Water')

        plt.tight_layout()
        plt.show()

