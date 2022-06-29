import numpy as np
import subprocess
import time
import matplotlib
from cycler import cycler
from numpy import float as r_dp

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

def build_polyfile(mesh_prefix,cpts,segs,holes,zones):
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
        x = cpts[i,0]
        y = cpts[i,1]
        flag = cpts[i,2]
        s = "{0:6.0F} {1:28.18E} {2:28.18E} {3:6.0F}\n"
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
        x = holes[i,0]
        y = holes[i,1]
        s = '{0:6.0F} {1:20.6E} {2:20.6E} 1\n'
        f1.write(s.format(i+1,x,y))
    f1.write('\n')

    #write the area constraints for zones
    f1.write('{0:6.0F}\n'.format(len(zones)))

    for i in range(len(zones)):
        x = zones[i,0]
        y = zones[i,1]
        area = zones[i,2]
        s = '{0:6.0F} {1:20.6E} {2:20.6E} {3:6.0F} {4:12.6E}\n'
        f1.write(s.format(i+1,x,y,i+1,area))

    f1.write('\n')
    f1.write('# triangle -pnq30Aae '+mesh_prefix+'.poly \n')
    f1.close()

    return


def call_triangle(mesh_prefix,triangle_path):
    command = [triangle_path+' -pnAae '+mesh_prefix+'.poly']
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


#modified from capsol GeneratateGrid
def generate_grid(**kwargs):
    #default inputs
    n = 1000 #number of grids in x
    m = 1000 #number of grids in z+
    l_js = 500 #number of grids in z-; gap between tip apex and sample surface not accounted
    h0 = r_dp(0.5) #[nm] #minimum grid size in x and z; fixed grid size in the gap
    rho_max = r_dp(20e6) #[nm] #box size in x
    z_max = r_dp(20e6) #[nm] #box size in z
    d_min = r_dp(20) #[nm] #minimum separation between tip apex and sample surface
    d_max = r_dp(20) #[nm] #maximum separation between tip apex and sample surface
    id_step = 2 #[nm] #istep (stepsize=istep*h0); z=z+idstep*h0 
    Hsam = r_dp(1e6) #[nm] #thickness_sample

    #update inputs according to kwargs
    for key,value in kwargs.items():
        if key=='n':
            n = value
        elif key=='m':
            m = value
        elif key=='l_js':
            l_js = value
        elif key=='h0':
            h0 = r_dp(value)
        elif key=='rho_max':
            rho_max = r_dp(value)
        elif key=='z_max':
            z_max = r_dp(value)
        elif key=='d_min':
            d_min = r_dp(value)
        elif key=='d_max':
            d_max = r_dp(value)
        elif key=='id_step':
            id_step = value
        elif key=='Hsam':
            Hsam = r_dp(value)

    r = np.zeros(n+1,dtype=r_dp) #x in[0,rho_max]
    hn = np.zeros(n+1,dtype=r_dp) #x in [0,rho_max]
    hm = np.zeros(m+1,dtype=r_dp) #z in[0,z_max]
    zm = np.zeros(m+1,dtype=r_dp) #z in[0,z_max]

    Nuni = 1
    for i in range(0,Nuni):
        r[i] = h0*i
        hn[i] = h0
    
    r[Nuni] = h0*Nuni
    
    qn = r_dp(1.012)
    # find the growth factor
    for qn in np.arange(1.0+1e-4,1.5+1e-4,1e-4,dtype=r_dp):
        x = h0*(1-qn**(n-Nuni))/(1-qn) #sum of geometric series
        if (x>=rho_max-r[Nuni]):
            break

    hn[Nuni] = h0*(rho_max-r[Nuni])/x
    r[Nuni+1] = np.sum(hn[0:Nuni+1])
    for i in range(Nuni+2,n+1):
        hn[i-1] = hn[i-2]*qn
        r[i] = np.sum(hn[0:i])
    hn[n] = hn[n-1]*qn

    #Z+ direction
    Nuni = 1
    hm[1:Nuni+1] = h0
    for j in range(1,Nuni+1):
        zm[j] = h0*j

    q = r_dp(1.010)
    for qm in np.arange(1.0+1e-4,1.5+1e-4,1e-4,dtype=r_dp):
        x = h0*(1-qm**(m-Nuni))/(1-qm)
        if (x>=z_max-zm[Nuni]):
            break

    hm[Nuni+1] = h0*(z_max-zm[Nuni])/x
    zm[Nuni+1] = np.sum(hm[1:Nuni+2])
    for j in range(Nuni+2,m+1):
        #print(j-1,hm[j-1])
        hm[j] = hm[j-1]*qm
        zm[j] = np.sum(hm[1:j+1])

    #Z- direction
    #Z=d_min where d_min is the minimum separation
    #Z=Z+idstep*h0 
    #js=-nint(Z/h0); l=l_js+(-js)
    #allocate (hn(0:n),r(0:n),hm(-l:m),zm(-l:m))
    z = r_dp(d_min) #vary with separation
    js = int(z/h0) #vary with separation
    l = l_js+js #l+js+int(z/h0), vary with separation

    hl = np.zeros(l+1,dtype=r_dp) #z in [0,separation+thickness_sample]
    zl = np.zeros(l+1,dtype=r_dp) #z in [0,separation,thickness_sample]
    hl[0:js+1] = h0
    for j in range(0,js+1):
        zl[j] = h0*j
    
    #hl[0] to hl[js-1] are fixed to h0; js increases with separation
    if l_js>0:
        q = r_dp(1.02)
        for ql in np.arange(1.0+1e-4,2.0+1e-4,1e-4,dtype=r_dp):
            x = h0*(1-ql**(l_js))/(1-ql)
            if (x>=Hsam):
                break
        hl[js] = h0*Hsam/x
    else:
        hl[js] = h0
    
    #when z increases from d_min to d_max
    #zm[zm>=0] part does not change
    #zm[(zm<0)&(zm>=-d_min)] part does not change
    #zm[zm<-d_min] part changes to include more gap(i.e. increased separation)
    #the number of grid between zm[zm<-d_min] and min(zm) are fixed
    for j in range(js+1,l+1):
        #print(hl[j],hl[j-1])
        hl[j] = hl[j-1]*ql
        zl[j] = np.sum(hl[0:j])
    
    #merge hm and hl to hm
    #merge zm and zl to zm
    hm = np.concatenate((np.flipud(hl),hm[1:]))
    zm = np.concatenate((-np.flipud(zl),zm[1:]))

    #np.sum(hn[:-1]) equal to r[-1]-r[0]
    #np.sum(hm[1:]) equal to zm[-1]-zm[0]
    return r,zm


#modified from capsol SetupProbe
def setup_probe(r,z,**kwargs):
    #z = z[z>=0]
    #default inputs
    Rtip = 20 #[nm]
    theta = 15 #[deg]
    Hcone = 15e3 #[nm]
    Rcant = 35e3 #[nm]
    dCant = 0.5e3 #[nm]
    
    #update inputs according to kwargs
    for key,value in kwargs.items():
        if key=='Rtip':
            Rtip = value
        elif key=='theta':
            theta = value
        elif key=='Hcone':
            Hcone = value
        elif key=='Rcant':
            Rcant = value
        elif key=='dCant':
            dCant = value

    #allocate (ProbeBot(0:n) , ProbeTop(0:n))
    n = len(r)-1
    m = len(z)-1
    theta = theta*np.pi/180
    Ra = Rtip*(1.0-np.sin(theta))
    Rc = Rtip*np.cos(theta)

    probe_bot = np.zeros(n,dtype=int)
    probe_top = np.zeros(n,dtype=int)
    j = np.where(z>=(Hcone+dCant))[0][0] #min index
    probe_top[:] = j
    
    nApex = 0
    nCone = 0
    nLever = 0
    nEdge = 0

    for i in range(n-1):
        x = r[i]
        if (x<Rc):
            nApex = i
            y = Rtip-np.sqrt(Rtip**2-x**2)
            j = np.where(z>=y)[0][0] #min index
            probe_bot[i] = j
            j = np.where(z>=(Hcone+dCant))[0][0] #min index
            probe_top[i] = j
        elif (x<((Hcone-Ra)*np.tan(theta)+Rc)):
            nCone = i
            y = (x-Rc)/np.tan(theta)+Ra
            j = np.where(z>=y)[0][0] #min index
            probe_bot[i] = j
            j = np.where(z>=(Hcone+dCant))[0][0] #min index
            probe_top[i] = j
        elif (x<Rcant):
            nLever = i
            y = Hcone
            j = np.where(z>=y)[0][0] #min index
            probe_bot[i] = j
            j = np.where(z>=(Hcone+dCant))[0][0] #min index
            probe_top[i] = j
        elif (x<(Rcant+dCant/2)):
            nEdge = i
            y = Hcone+dCant/2-np.sqrt((dCant/2)**2-(x-Rcant)**2)
            j = np.where(z>=y)[0][0] #min index
            probe_bot[i] = j
            j = np.where(z>=(y+2*np.sqrt((dCant/2)**2-(x-Rcant)**2)))[0][0] #min index
            probe_top[i] = j

    #fix probe_top if it overlaps with probe_bot
    if max(probe_bot)>=min(probe_top):
        probe_top[:] = max(probe_bot)+1

    probe = np.zeros((0,5))
    #ind = np.arange(1,nApex+1,1)
    ind = np.arange(0,nApex+1,1)
    flag = np.ones_like(ind)*1
    probe = np.r_[probe,np.c_[r[ind],z[probe_bot[ind]],ind,probe_bot[ind],flag]]
    ind = np.arange(nApex+1,nCone+1,1)
    flag = np.ones_like(ind)*2
    probe = np.r_[probe,np.c_[r[ind],z[probe_bot[ind]],ind,probe_bot[ind],flag]]
    ind = np.arange(nCone+1,nLever+1,1)
    flag = np.ones_like(ind)*3
    probe = np.r_[probe,np.c_[r[ind],z[probe_bot[ind]],ind,probe_bot[ind],flag]]
    ind = np.arange(nLever+1,nEdge+1,1)
    flag = np.ones_like(ind)*4
    probe = np.r_[probe,np.c_[r[ind],z[probe_bot[ind]],ind,probe_bot[ind],flag]]
    ind = np.flipud(np.arange(nLever+1,nEdge+1,1))
    flag = np.ones_like(ind)*4
    probe = np.r_[probe,np.c_[r[ind],z[probe_top[ind]],ind,probe_top[ind],flag]]
    ind = np.flipud(np.arange(nCone+1,nLever+1,1))
    flag = np.ones_like(ind)*3
    probe = np.r_[probe,np.c_[r[ind],z[probe_top[ind]],ind,probe_top[ind],flag]]
    ind = np.flipud(np.arange(nApex+1,nCone+1,1))
    flag = np.ones_like(ind)*2
    probe = np.r_[probe,np.c_[r[ind],z[probe_top[ind]],ind,probe_top[ind],flag]]
    #ind = np.flipud(np.arange(1,nApex+1,1))
    ind = np.flipud(np.arange(0,nApex+1,1))
    flag = np.ones_like(ind)*1
    probe = np.r_[probe,np.c_[r[ind],z[probe_top[ind]],ind,probe_top[ind],flag]]

    fmt = '%15.5e %14.5e %5d %5d %5d'
    header='  # rho, z,   i,  j,  code:  1=sphere, 2=cone, 3=cant, 4= edge'
    comments=''
    np.savetxt('probe.out',probe,fmt,header=header,comments='')
    return probe


#similar to setup_probe but to setup aw/sw interface
def setup_hline(r,z,z0):
    n = len(r)
    line_top = np.zeros(n,dtype=int)+np.where(z>=z0)[0][0]

    ind = np.arange(0,n)
    line = np.c_[r[ind],z[line_top[ind]],ind,line_top[ind]]
    return line

#similar to setup_probe but to setup outermost box
def setup_box(r,z):
    n = len(r)
    m = len(z)

    box_bot = np.zeros(n,dtype=int)+np.argmin(z)
    box_top = np.zeros(n,dtype=int)+np.argmax(z)

    box_ls = np.zeros(m,dtype=int)+np.argmin(r)
    box_rs = np.zeros(m,dtype=int)+np.argmax(r)

    box = [None]*4
    ind = np.arange(n)
    box[0] = np.c_[r[ind],z[box_bot[ind]],ind,box_bot[ind]]
    ind = np.arange(m)
    box[1] = np.c_[r[box_rs[ind]],z[ind],box_rs[ind],ind]
    ind = np.flipud(np.arange(n))
    box[2] = np.c_[r[ind],z[box_top[ind]],ind,box_top[ind]]
    ind = np.flipud(np.arange(m))
    box[3] = np.c_[r[box_ls[ind]],z[ind],box_ls[ind],ind]

    return box


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
        build_polyfile(mesh.prefix,mesh.cpts,mesh.segs,mesh.holes,mesh.zones)
        call_triangle(mesh.prefix,mesh.triangle)
        mesh.dist_factor = 1.0
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
        #input: f_n.shape (n_node,)
        #output: f_out.shape (n_elem,3) with f_e/f_x/f_y
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
        #input: f_x.shape (n_elem,)
        #input: f_y.shape (n_elem,)
        #output: flux.shape (n_edge,2)
        flux = np.zeros((len(self.edges),2),dtype=f_x.dtype)
        for i in range(2): #possible two adjacent elements
            nodes = self.nodes
            mask = self.edge_to_elem[:,i]>=0

            #vector u points from edges[:,0] to edges[:,1]
            edges = self.edges[mask,:]
            u_x = nodes[edges[:,1],0]-nodes[edges[:,0],0] #(n_edge,)
            u_y = nodes[edges[:,1],1]-nodes[edges[:,0],1] #(n_edge,)
            u_len = np.sqrt(u_x**2+u_y**2) #(n_edge,)
            u_x = u_x/u_len
            u_y = u_y/u_len

            #vector v points from elem_mids to edges[:,0]
            elements = self.elements[self.edge_to_elem[mask,i],:] #(n_edge,3)
            elem_mids = self.elem_mids[self.edge_to_elem[mask,i],:] #(n_edge,2)
            v_x = nodes[edges[:,0],0]-elem_mids[:,0] #(n_edge,)
            v_y = nodes[edges[:,0],1]-elem_mids[:,1] #(n_edge,)
            #v_len = np.sqrt(v_x**2+v_y**2)
            #v_x = v_x/v_len
            #v_y = v_y/v_len

            #vector n = v-dot(u,v)u
            n_x = v_x-(u_x*v_x+u_y*v_y)*u_x #(n_edge,)
            n_y = v_y-(u_x*v_x+u_y*v_y)*u_y #(n_edge,)
            n_len = np.sqrt(n_x**2+n_y**2) #(n_edge,)
            n_x = n_x/n_len
            n_y = n_y/n_len
            #print(np.r_[n_x,n_y])

            try:
                flux[mask,i] = (f_x[self.edge_to_elem[mask,i]]*n_x
                                +f_y[self.edge_to_elem[mask,i]]*n_y) #wrapped
            except:
                flux[mask,i] = f_x*n_x+f_y*n_y #for test purpose f_x=1,f_y=1

        return flux

    def force2d(self,u):
        #input: f_x.shape (n_elem,) equal to -e_x
        #input: f_y.shape (n_elem,) equal to -e_y
        #output: force.shape (n_edge,2) -> f_out (n_edge,3)[mask,:]
#         mask = self.is_with_equipotential
#         edges = self.edges[mask,:]
#         edge_mids = self.edge_mids[mask,:]
#         f_out = np.zeros((len(edges),5),dtype=u.dtype)

#         x = mesh_grid[0]
#         y = mesh_grid[1]
#         is_on_probe = np.zeros((len(self.nodes),4),dtype=bool)
        probe = setup_probe(self.grid_x,self.grid_y)
        n_bot = len(probe)//2
        f_out = np.zeros((n_bot,5))
        for i in range(n_bot):
            x0 = self.grid_x[probe[i,2].astype(int)]
            y0 = self.grid_y[probe[i,3].astype(int)]
            x1 = min(self.grid_x[self.grid_x>x0])
            y1 = max(self.grid_y[self.grid_y<y0])
            y2 = max(self.grid_y[self.grid_y<y1])
#             print('i',i)
#             print(x0,x1)

            dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y0)**2
            ind_0 = np.argmin(dist2)
            x_0 = self.nodes[ind_0,0]
            y_0 = self.nodes[ind_0,1]
#             print(ind_0,x0,y0)
            
            dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y1)**2
            ind_1 = np.argmin(dist2)
            x_1 = self.nodes[ind_1,0]
            y_1 = self.nodes[ind_1,1]
            u_1 = u[ind_1]
#             print(ind_1,x0,y1)
#             print(x_0,x_1)
            
            dist2 = (self.nodes[:,0]-x1)**2+(self.nodes[:,1]-y1)**2
            ind_2 = np.argmin(dist2)
            x_2 = self.nodes[ind_2,0]
            y_2 = self.nodes[ind_2,1]
            u_2 = u[ind_2]

            dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y2)**2
            ind_3 = np.argmin(dist2)
            x_3 = self.nodes[ind_3,0]
            y_3 = self.nodes[ind_3,1]
            u_3 = u[ind_3]

            #compute f_out
            hn = x_2-x_1
            hm = y_1-y_3
            ex = -(u_2-u_1)/hn #skip stepwise boundary by not using u_0
            ey = -(u_1-u_3)/hm #skip stepwise boundary by not using u_0
            e2 = ex**2+ey**2
            df = 0.5*(x_0+x_1)*hn*e2

            f_out[i,0] = 0.5*(x_0+x_1) #x
            f_out[i,1] = y_0 #y
            f_out[i,2] = 0.5*(x_0+x_1)*e2*hn

#         for i in range(len(edges)):
#             ind = np.argmin(self.nodes[edges[i,:],0]) #left point
#             is_on_probe[edges[i,ind],0] = True
#             x0 = self.nodes[edges[i,ind],0]
#             y0 = self.nodes[edges[i,ind],1]
#             x1 = min(x[x>x0])
#             y1 = max(y[y<y0])
#             y2 = max(y[y<y1])

#             dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y1)**2
#             n_ind = np.argmin(dist2)
#             is_on_probe[n_ind,1] = True
#             u1 = u[n_ind]

#             dist2 = (self.nodes[:,0]-x1)**2+(self.nodes[:,1]-y1)**2
#             n_ind = np.argmin(dist2)
#             is_on_probe[n_ind,2] = True
#             u2 = u[n_ind]

#             dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y2)**2
#             n_ind = np.argmin(dist2)
#             is_on_probe[n_ind,3] = True
#             u3 = u[n_ind]
            
#             #start computing
#             hn = x1-x0
#             hm = y0-y1
#             ex = -(u2-u1)/hn
#             ey = -(u1-u3)/hm
#             e2 = ex**2+ey**2
#             df = 0.5*(x0+x1)*hn*e2

#             f_out[i,0] = 0.5*(x0+x1) #x
#             f_out[i,1] = y0 #y
#             f_out[i,2] = 0.5*(x0+x1)*e2*hn
        
#         mask = edge_mids[:,0]>20 #test only!
#         ztop = max(edge_mids[mask,1])
#         zbot = min(edge_mids[mask,1])
#         print(ztop,zbot)
#         mask = edge_mids[:,1]<0.5*(ztop+zbot)
#         f_out = f_out[mask,:]

#         self.is_on_probe = is_on_probe
        return f_out

#         if 'is_on_probe' not in self.__dict__.keys():
#             x = mesh_grid[0]
#             y = mesh_grid[1]
#             is_on_probe = np.zeros((len(self.nodes),4),dtype=bool)
#             for i in range(len(edges)):
#                 ind = np.argmin(self.nodes[edges[i,:],0]) #left point
#                 is_on_probe[edges[i,ind],0] = True
#                 x0 = self.nodes[edges[i,ind],0]
#                 y0 = self.nodes[edges[i,ind],1]
#                 x1 = min(x[x>x0])
#                 y1 = max(y[y<y0])
#                 y2 = max(y[y<y1])

#                 dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y1)**2
#                 n_ind = np.argmin(dist2)
#                 is_on_probe[n_ind,1] = True

#                 dist2 = (self.nodes[:,0]-x1)**2+(self.nodes[:,1]-y1)**2
#                 n_ind = np.argmin(dist2)
#                 is_on_probe[n_ind,2] = True

#                 dist2 = (self.nodes[:,0]-x0)**2+(self.nodes[:,1]-y2)**2
#                 n_ind = np.argmin(dist2)
#                 is_on_probe[n_ind,3] = True
#             self.is_on_probe = is_on_probe

#         x0 = self.nodes[self.is_on_probe[:,0],0]
#         x1 = self.nodes[self.is_on_probe[:,2],0]
#         hn = x1-x0

#         y0 = self.nodes[self.is_on_probe[:,0],1]
#         y1 = self.nodes[self.is_on_probe[:,1],1]
#         hm = y0-y1

#         u1 = u[self.is_on_probe[:,1]] #under tip
#         u2 = u[self.is_on_probe[:,2]] #under tip and move+1 along x
#         u3 = u[self.is_on_probe[:,3]] #under tip and move-1 along y
#         ex = -(u2-u1)/hn
#         ey = -(u1-u3)/hm
#         e2 = ex**2+ey**2

#         f_out[:,0] = 0.5*(x0+x1) #x
#         f_out[:,1] = y0 #y
# #         f_out[:,2] = e2
#         f_out[:,2] = 0.5*(x0+x1)*e2
        
#         ztop = max(edge_mids[:,1])
#         mask = edge_mids[:,1]<ztop
#         f_out = f_out[mask,:]

#         return f_out

#             x1 = self.nodes[self.is_on_probe[:,0],0]
#             y1 = self.nodes[self.is_on_probe[:,1],1]
#             u0 = u[self.is_on_probe[:,0]]

#             x1 = self.nodes[self.is_on_probe[:,2]]
#             hn = self.nodes[#x1-x0


            #print(x_i,self.nodes[is_on_probe[:,0],0])
            #print(y2,y1,y0)
            #print(x0,x1)

#         nodes = self.nodes
#         elements = self.elements
#         elem_mids = self.elem_mids
#         elem_factor = self.elem_factor

#         mask = self.is_with_equipotential
#         edges = self.edges[mask,:]
#         edge_mids = self.edge_mids[mask,:]
#         edge_len = self.edge_len[mask]
#         edge_factor = self.edge_factor[mask]

#         if 'is_in_probe' not in self.__dict__.keys():
#             is_in_probe = np.zeros(len(elements),dtype=bool) #elements next to probe
#             x = elem_mids[:,0]
#             y = elem_mids[:,1]
#             for i in range(len(edges)):
#                 edge_x = edge_mids[i,0]
#                 edge_y = edge_mids[i,1]
#                 j = np.argmin((x-edge_x)**2+(y-edge_y)**2)
#                 is_in_probe[j] = True
#             self.is_in_probe = is_in_probe

#         #vector u points from edges[:,0] to edges[:,1]
#         u_x = nodes[edges[:,1],0]-nodes[edges[:,0],0] #(n_edge,)[mask]
#         u_y = nodes[edges[:,1],1]-nodes[edges[:,0],1] #(n_edge,)[mask]
#         u_len = np.sqrt(u_x**2+u_y**2) #(n_edge,)[mask]
#         u_x = u_x/u_len
#         u_y = u_y/u_len

#         #vector v points from elem_mids to edges[:,0]
#         #elements = self.elements[self.is_in_probe,:] #(n_edge,3)[mask,:]
#         #elem_mids = self.elem_mids[self.is_in_probe,:] #(n_edge,2)[mask,:]
#         v_x = nodes[edges[:,0],0]-elem_mids[self.is_in_probe,0] #(n_edge,)[mask]
#         v_y = nodes[edges[:,0],1]-elem_mids[self.is_in_probe,1] #(n_edge,)[mask]
#         #v_len = np.sqrt(v_x**2+v_y**2)
#         #v_x = v_x/v_len
#         #v_y = v_y/v_len

#         #vector n = v-dot(u,v)u
#         n_x = v_x-(u_x*v_x+u_y*v_y)*u_x #(n_edge,)[mask]
#         n_y = v_y-(u_x*v_x+u_y*v_y)*u_y #(n_edge,)[mask]
#         n_len = np.sqrt(n_x**2+n_y**2) #(n_edge,)[mask]
#         n_x = n_x/n_len
#         n_y = n_y/n_len
#         #print(np.c_[n_x,n_y,n_x*u_x+n_y*u_y])
#         #print(np.c_[u_x*edge_len,n_y*edge_len])

#         if False:
#             e2 = (f_x[self.is_in_probe]**2
#                  +f_y[self.is_in_probe]**2)
#             hn = abs(nodes[edges[:,1],0]-nodes[edges[:,0],0])
#             zm = np.minimum(nodes[edges[:,0],1],nodes[edges[:,1],1])
#             f_out = np.zeros((len(edges),5),dtype=f_x.dtype)
#             f_out[:,0] = edge_mids[:,0] #x
#             #f_out[:,1] = edge_mids[:,1] #y
#             f_out[:,1] = zm
#             f_out[:,2] = e2*hn*edge_factor*np.sign(n_y) #force by element
#             f_out[:,3] = e2*np.sign(n_y)
#             f_out[:,4] = edge_factor*np.sign(n_y)
#         else:
#             e2 = (f_x[self.is_in_probe]*n_x+f_y[self.is_in_probe]*n_y)**2
#             hn = abs(nodes[edges[:,1],0]-nodes[edges[:,0],0])
#             zm = np.minimum(nodes[edges[:,0],1],nodes[edges[:,1],1])
#             f_out = np.zeros((len(edges),5),dtype=f_x.dtype)
#             f_out[:,0] = edge_mids[:,0] #x
#             #f_out[:,1] = edge_mids[:,1] #y
#             f_out[:,1] = zm
#             f_out[:,2] = e2*n_y*edge_len*edge_factor #force by element
#             f_out[:,3] = e2*np.sign(n_y)

#         return f_out

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

    def find_edge_neigh(self,edge_mask):
        print('Finding adjacent elements for line segments')
        print('This will take a while')
        start = time.time()
        n_elem = len(self.elements)
        n_edge = len(self.edges)
        edge_to_elem = np.zeros((n_edge,2),dtype=int)-1
        elem_to_edge = [[]*1 for i in range(n_elem)]
        
        edge_ind = np.where(edge_mask)[0]
        for i in edge_ind:
            mask_1 = self.edges[i,0]==self.elements[:,0]
            mask_1 = mask_1|(self.edges[i,0]==self.elements[:,1])
            mask_1 = mask_1|(self.edges[i,0]==self.elements[:,2])

            mask_2 = self.edges[i,1]==self.elements[:,0]
            mask_2 = mask_2|(self.edges[i,1]==self.elements[:,1])
            mask_2 = mask_2|(self.edges[i,1]==self.elements[:,2])

            ind = np.where(mask_1&mask_2)[0]
            if len(ind)==1:
                edge_to_elem[i,0] = ind[0]
                elem_to_edge[ind[0]].append(i)
            elif len(ind)==2:
                edge_to_elem[i,0] = ind[0]
                edge_to_elem[i,1] = ind[1]
                elem_to_edge[ind[0]].append(i)
                elem_to_edge[ind[1]].append(i)

        #elem_to_edge = np.array(elem_to_edge,dtype=int)
        #sort of self.edge_to_elem by columns
        #left column: flux2d positive if (f_x,f_y) is (1,1)
        #right column: flux2d negative if (f_x,f_y) is (1,1)
        elapsed = time.time()-start
        print('Time elapsed ',elapsed,'sec')
        print('')
        return edge_to_elem
    
    def find_elem_neigh(self):
        #search by nodes may be more efficient
        elem_to_edge = []
        return elem_to_edge

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
        ax.triplot(x,y,self.elements[:,:],linewidth=0.2,color='tab:blue')
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        if self.axis_symmetry=='X':
            ax.triplot(x,-y,self.elements[:,:],linewidth=0.2,color='tab:blue')
        
        if self.axis_symmetry=='Y':
            ax.triplot(-x,y,self.elements[:,:],linewidth=0.2,color='tab:blue')

        #plot specified elements
        for i in range(len(elem_flags)):
            mask = self.elem_flags==elem_flags[i]
            if np.sum(mask)>0:
                ax.triplot(x,y,self.elements[mask,:],linewidth=0.2,
                           color='tab:orange') #wrapped

#         #plot all edges in the background
#         x = self.nodes[self.edges,0]
#         y = self.nodes[self.edges,1]
#         for attr in Flags.__dict__.keys():
#             if not '__' in attr:
#                 mask = self.edge_flags==Flags.__dict__[attr]
#                 ax.plot(x[mask,:].T,y[mask,:].T,color='tab:orange',alpha=0.2)

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
        mask = self.is_with_stern|self.is_with_equipotential
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
        print('This will take a while')
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
        print('This will take a while')
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
        radius_air = r_dp(self.radius_air) #radius of the air
        height_air = r_dp(self.height_air) #height of the air
        height_gap = r_dp(self.height_gap) #gap between tip apex and sample surface
        height_water = r_dp(self.height_water) #thickness of thin water film
        height_solid = r_dp(self.height_air) #height of the solid

        #user inputs of probe geometry
        radius_tip = r_dp(self.radius_tip) #radius of probe tip
        radius_cone = r_dp(self.radius_cone) #radius of probe cone
        height_cone = r_dp(self.height_cone) #height of probe cone
        radius_disk = r_dp(self.radius_disk) #radius of probe disk
        height_disk = r_dp(self.height_disk) #height of probe disk

        #user inputs of mesh discretization
        area_air = r_dp(self.area_air)
        area_water = r_dp(self.area_water)
        area_solid = r_dp(self.area_solid)

        #discretize rho
        #lambda_d = min(9e-9,height_water)
        #rho = discretize_rho(lambda_d,rho_min=0,rho_max=radius_solid)
        #skip refinement between 0 and radius_solid
        rho = np.r_[r_dp(0),radius_air]

        #insert air-water interface into the discretization
        #mask = rho<height_water
        #rho = np.r_[rho[mask],height_water,rho[~mask]]
        #ind = np.argmin((rho-height_water)**2)
        #rho[ind] = height_water

        #insert height_gap into the discretization
        #mask = rho<height_gap
        #rho = np.r_[rho[mask],height_gap,rho[~mask]]

        #print out discretization
        #print('See radial discretization below')
        #print(rho)

        #generate mesh
        #This script only works for the same height and radius for air and solid
        #X is the axis of symmetry
        #Y is the longitudinal axis
        cpts = np.zeros((0,3),dtype=r_dp) #coord_x/coord_y/flag of control points
        segs = np.zeros((0,3)) #ind_a/ind_b/flag of line segmenets
        holes = np.zeros((0,2),dtype=r_dp) #coord_x/coord_y
        zones = np.zeros((0,3),dtype=r_dp) #coord_x/coord_y/area

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #define the lowermost, rightmost, and topmost boundary points
        #radius_b = max(radius_air,radius_solid)
        #height_b = max(height_air,height_solid)
        x = np.r_[r_dp(0),radius_air,radius_air,r_dp(0)]
        y = np.r_[-height_solid-height_gap,-height_solid-height_gap,
                  height_air,height_air]
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*1]] #node flag of 1

        #define the leftmost boundary points (on the axis of symmetry)
        #skip points at two ends (top and bottom)
        mask = (rho>0)&(rho<height_gap)
        y = np.r_[-np.flipud(rho[:-1])-height_gap,rho[mask]-height_gap,
                  height_water-height_gap,0.0,2*radius_tip,height_cone,
                  height_cone+height_disk]
        x = np.zeros_like(y,dtype=r_dp)
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*1]] #node flag of 1

        #-----------------------------------------------------------------------
        #define the top edge points of the solid
        #skip edge points on the axis of symmetry
        x = np.r_[rho[1:-1],radius_air]
        y = np.zeros_like(x,dtype=r_dp)-height_gap
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the top edge points of the water
        #skip edge points on the axis of symmetry
        x = np.r_[rho[1:-1],radius_air]
        y = np.zeros_like(x,dtype=r_dp)+height_water-height_gap
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the side edge points on the tip surface
        #skip edge points on the axis of symmetry
        nA = 32
        dA = np.pi/nA
        phi = np.arange(1,nA)*dA-np.pi/2 #half the circle
        x = radius_tip*np.cos(phi)
        y = radius_tip*np.sin(phi)+radius_tip #so that tip apex sits at Y=0
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the side edge points on the cone surface
        x = np.r_[radius_cone]
        y = np.r_[height_cone]
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #-----------------------------------------------------------------------
        #define the side edge points along the cantilever
        x = np.r_[radius_disk,radius_disk]
        y = np.r_[0,height_cone+height_disk]
        cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #define the segments on the lowermost boundary
        x = np.r_[0,radius_air]
        y = np.r_[-height_solid-height_gap,-height_solid-height_gap]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.bottom_bound]]

        #define the segments on the rightmost boundary
        x = np.r_[radius_air,radius_air]
        y = np.r_[-height_solid-height_gap,height_air]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.right_bound]]

        #define the segments on the topmost boundary
        x = np.r_[0,radius_air]
        y = np.r_[height_air,height_air]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.top_bound]]

        #define the segments on the leftmost boundary (axis of symmetry)
        mask = (rho>0)&(rho<height_gap)
        y = np.r_[-height_solid-height_gap,-np.flipud(rho[:-1])-height_gap,
                  rho[mask]-height_gap,height_water-height_gap,0.0,
                  2*radius_tip,height_cone,height_cone+height_disk,height_air]
        x = np.zeros_like(y)
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.axis_symmetry]]

        #-----------------------------------------------------------------------
        #define the segments on the top edge of the solid (solid-liquid interface)
        x = np.r_[rho[:-1],radius_air]
        y = np.zeros_like(x)-height_gap
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.sw_interface]]

        #-----------------------------------------------------------------------
        #define the segments on the top edge of the water
        x = np.r_[rho[:-1],radius_air]
        y = np.zeros_like(x)+height_water-height_gap
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.aw_interface]]

        #-----------------------------------------------------------------------
        #define the segments along the bottom tip surface
        nA = 32
        dA = np.pi/nA
        phi = np.arange(0,nA//2+1)*dA-np.pi/2 #half the circle
        #print(phi*180/np.pi)
        x = radius_tip*np.cos(phi)
        y = radius_tip*np.sin(phi)+radius_tip
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

#         #define the segments along the top tip surface
#         nA = 32
#         ns = nA+1-2
#         dA = np.pi/nA
#         phi = np.arange(nA//2,nA+1)*dA-np.pi/2 #half the circle
#         #print(phi*180/np.pi)
#         x = radius_tip*np.cos(phi)+0.0
#         y = radius_tip*np.sin(phi)+height_gap+radius_tip
#         for i in range(len(x)-1):
#             ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
#             ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
#             segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

        #-----------------------------------------------------------------------
        #define the bottom segments along the cone surface
        x = np.r_[radius_tip,radius_cone]
        y = np.r_[radius_tip,height_cone]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

#         #define the top segments along the cone surface
#         x = np.r_[0,radius_cone]
#         y = np.r_[height_cone,height_cone]+height_gap+2*radius_tip
#         for i in range(len(x)-1):
#             ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
#             ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
#             segs = np.r_[segs,np.c_[ind_a,ind_b,100]]

        #-----------------------------------------------------------------------
        #define the remaining segments along the cantilever surface
        x = np.r_[0,radius_disk,radius_disk,radius_cone]
        y = np.r_[height_disk,height_disk,0,0]+height_cone
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]]

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #use mesh_grid to define control points and line segments
        if 'mesh_grid' in self.__dict__.keys():
            cpts = np.zeros((0,3))
            segs = np.zeros((0,3))

            x = self.mesh_grid[0]
            y = self.mesh_grid[1]

            #x = x[x<=radius_air]
            #y = y[(y>=-height_solid-height_gap)&(y<=height_air)]

            xg,yg = np.meshgrid(x,y)
            xg = xg.ravel()
            yg = yg.ravel()
            cpts = np.c_[xg,yg,np.zeros_like(xg)]

            probe = setup_probe(x,y)
            ind_a = probe[:-1,2]+probe[:-1,3]*len(x)
            ind_b = probe[1:,2]+probe[1:,3]*len(x)
            segs = np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                         +Flags.equipotential_surf]
            
            line = setup_hline(x,y,-height_gap)
            ind_a = line[:-1,2]+line[:-1,3]*len(x)
            ind_b = line[1:,2]+line[1:,3]*len(x)
            segs = np.r_[segs,np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                                    +Flags.sw_interface]]

            line = setup_hline(x,y,height_water-height_gap)
            ind_a = line[:-1,2]+line[:-1,3]*len(x)
            ind_b = line[1:,2]+line[1:,3]*len(x)
            segs = np.r_[segs,np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                                    +Flags.aw_interface]]

            box = setup_box(x,y)
            ind_a = box[0][:-1,2]+box[0][:-1,3]*len(x)
            ind_b = box[0][1:,2]+box[0][1:,3]*len(x)
            segs = np.r_[segs,np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                                    +Flags.bottom_bound]]
            ind_a = box[1][:-1,2]+box[1][:-1,3]*len(x)
            ind_b = box[1][1:,2]+box[1][1:,3]*len(x)
            segs = np.r_[segs,np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                                    +Flags.right_bound]]
            ind_a = box[2][:-1,2]+box[2][:-1,3]*len(x)
            ind_b = box[2][1:,2]+box[2][1:,3]*len(x)
            segs = np.r_[segs,np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                                    +Flags.top_bound]]
            ind_a = box[3][:-1,2]+box[3][:-1,3]*len(x)
            ind_b = box[3][1:,2]+box[3][1:,3]*len(x)
            segs = np.r_[segs,np.c_[ind_a,ind_b,np.zeros_like(ind_a)
                                    +Flags.axis_symmetry]]            

            #update the background and probe geometry
            mask = segs[:,-1]==Flags.sw_interface
            z_sw = cpts[segs[mask,:-1].astype(int),1][0][0]
            mask = segs[:,-1]==Flags.aw_interface
            z_aw = cpts[segs[mask,:-1].astype(int),1][0][0]

            radius_air = max(x)
            height_air = max(y)
            height_gap = abs(z_sw)
            height_water = height_gap-abs(z_aw)
            height_solid = abs(min(y))-height_water

#             self.radius_air = radius_air
#             self.height_air = height_air
#             self.height_gap = height_gap
#             self.height_water = height_water
#             self.height_solid = height_solid
            
            radius_tip = max(probe[probe[:,-1]==1,0])
            radius_cone = max(probe[probe[:,-1]==2,0])
            radius_disk = max(probe[probe[:,-1]==3,0])
            mask = probe[:,-1]==3
            height_disk = max(probe[mask,1])-min(probe[mask,1])
            height_cone = max(probe[probe[:,-1]==2,1])-height_disk

#             self.radius_tip = radius_tip
#             self.radius_cone = radius_cone
#             self.height_cone = height_cone
#             self.radius_disk = radius_disk
#             self.height_disk = height_disk

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #define markers for holes and zones
#         x = np.r_[0,0,0,0,0,0]+radius_tip/2 #solid,water,tip,cone,arm,air
#         y = np.r_[-height_solid/2,height_water/2,height_gap+radius_tip,
#                   height_gap+2*radius_tip+height_cone/2,
#                   height_gap+2*radius_tip+height_cone+height_disk/2,
#                   height_air/4]
#         area = np.r_[1,1,1,100,100,900]
#         zones = np.r_[zones,np.c_[x,y,area]]
        x = np.r_[radius_tip/2] #cantilever tip
        y = np.r_[height_cone/2]
        holes = np.r_[holes,np.c_[x,y]]

        x = np.r_[0,0,0]+radius_air*0.99 #solid,water,air
        y = np.r_[-height_solid*0.99,height_water/2-height_gap,height_air*0.99]
        area = np.r_[area_solid,area_water,area_air]
        zones = np.r_[zones,np.c_[x,y,area]]

        self.cpts = cpts
        self.segs = segs
        self.holes = holes
        self.zones = zones

        #***********************************************************************
        #-----------------------------------------------------------------------
        #=======================================================================
        #write poly file and call triangle
        if 'dist_factor' in self.__dict__.keys():
            cpts[:,0] = cpts[:,0]*self.dist_factor
            cpts[:,1] = cpts[:,1]*self.dist_factor
            zones[:,2] = zones[:,2]*self.dist_factor**2 

        if self.build_mesh:
            build_polyfile(self.mesh_prefix,cpts,segs,holes,zones)
            call_triangle(self.mesh_prefix,'triangle')

    def triplot(self,xunit=[],yunit=[]):
        radius_air = self.radius_air
        radius_tip = self.radius_tip
        radius_disk = self.radius_disk
        height_water = self.height_water
        height_gap = self.height_gap

        if xunit=='nanometer' or xunit=='nm':
            xunit = 'nm'
        elif xunit=='micrometer' or xunit=='um':
            xunit = '$\mu$m'
        elif xunit=='millimeter' or xunit=='mm':
            xunit = 'mm'
        else:
            xunit = 'm'

        if yunit=='nanometer' or yunit=='nm':
            yunit = 'nm'
        elif yunit=='micrometer' or yunit=='um':
            yunit = '$\mu$m'
        elif yunit=='millimeter' or yunit=='mm':
            yunit = 'mm'
        else:
            yunit = 'm'

        nodes,node_flags = import_nodes(self.mesh_prefix)
        elements,elem_flags = import_elements(self.mesh_prefix)
        edges,edge_flags = import_edges(self.mesh_prefix)
        print('THE NUMBER OF NODES IS: %d'%len(nodes))
        print('THE NUMBER OF ELEMENTS IS: %d'%len(elements))
        print('THE NUMBER OF EDGES IS: %d'%len(edges))
        print('node_flags',np.unique(node_flags))
        print('elem_flags',np.unique(elem_flags))
        print('edge_flags',np.unique(edge_flags))
        print('')

        x = nodes[:,0]
        y = nodes[:,1]

        fig,ax=plt.subplots(2,2,figsize=(8,8),dpi=80)
        axs=ax.flatten()

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[0].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[0].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[0].set_xlabel('X ($\mu$m)')
        axs[0].set_ylabel('Y ($\mu$m)')
        axs[0].set_aspect('equal')
        axs[0].set_xlim(-radius_air*1.1,radius_air*1.1)
        axs[0].set_ylim(-radius_air*1.1,radius_air*1.1)
        axs[0].set_title('Zoom Out')

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[1].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[1].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        mask=(elem_flags>=1)&(elem_flags<=5)
        axs[1].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:orange')
        axs[1].set_xlabel('X ($\mu$m)')
        axs[1].set_ylabel('Y ($\mu$m)')
        axs[1].set_aspect('equal')
        axs[1].set_xlim(-radius_disk*1.1,radius_disk*1.1)
        axs[1].set_ylim(-radius_disk*1.1,radius_disk*1.1)
        axs[1].set_title('Zoom In: Cantilever')

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[2].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[2].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        mask=(elem_flags>=1)&(elem_flags<=5)
        axs[2].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:orange')
        axs[2].set_xlabel('X ($\mu$m)')
        axs[2].set_ylabel('Y ($\mu$m)')
        axs[2].set_aspect('equal')
        axs[2].set_xlim(-(radius_tip*4+height_gap),(radius_tip*4+height_gap))
        axs[2].set_ylim(-(radius_tip*4+height_gap),(radius_tip*4+height_gap))
        axs[2].set_title('Zoom In: Tip')

        mask=(elem_flags<=3)|(elem_flags>=4)
        axs[3].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        axs[3].triplot(-x,y,elements[mask,:],linewidth=0.2,color='tab:blue')
        mask=(elem_flags>=2)&(elem_flags<=2)
        axs[3].triplot(x,y,elements[mask,:],linewidth=0.2,color='tab:orange')
        axs[3].set_xlabel('X ($\mu$m)')
        axs[3].set_ylabel('Y ($\mu$m)')
        axs[3].set_aspect('equal')
        axs[3].set_xlim(-height_water*10,height_water*10)
        axs[3].set_ylim(-height_water*10,height_water*10)
        axs[3].set_title('Zoom In: Water')

        plt.tight_layout()
        plt.show()
    
    def visualize(self,cpt_flags='all',seg_flags='all',xunit=[],yunit=[]):
        if cpt_flags=='all':
            cpt_flags = np.unique(self.cpts[:,-1])
            #cpt_flags = np.reshape(cpt_flags,(1,-1))
        elif cpt_flags=='none':
            cpt_flags = []

        if seg_flags=='all':
            seg_flags = np.unique(self.segs[:,-1])
            #seg_flags = np.reshape(seg_flags,(1,-1))
        elif seg_flags=='none':
            seg_flags = []

        mask = np.zeros(len(self.cpts),dtype=bool)
        for i in range(len(cpt_flags)):
            mask = mask|(self.cpts[:,-1]==cpt_flags[i])
        cpts = self.cpts[mask,:]

        mask = np.zeros(len(self.segs),dtype=bool)
        for i in range(len(seg_flags)):
            mask = mask|(self.segs[:,-1]==seg_flags[i])
        segs = self.segs[mask,:]

#         if len(cpts)==0:
#             cpts = np.zeros((0,3))
#         else:
#             cpts = self.cpts

#         if len(segs)==0:
#             segs = np.zeros((0,3))
#         else:
#             segs = self.segs

        x = self.cpts[segs[:,:-1].astype(int),0]
        y = self.cpts[segs[:,:-1].astype(int),1]

        radius_air = self.radius_air
        radius_tip = self.radius_tip
        radius_disk = self.radius_disk
        height_water = self.height_water
        height_gap = self.height_gap
        
        if xunit=='nanometer' or xunit=='nm':
            xunit = 'nm'
        elif xunit=='micrometer' or xunit=='um':
            xunit = '$\mu$m'
        elif xunit=='millimeter' or xunit=='mm':
            xunit = 'mm'
        else:
            xunit = 'm'
        
        if yunit=='nanometer' or yunit=='nm':
            yunit = 'nm'
        elif yunit=='micrometer' or yunit=='um':
            yunit = '$\mu$m'
        elif yunit=='millimeter' or yunit=='mm':
            yunit = 'mm'
        else:
            yunit = 'm'
        
        fig,ax = plt.subplots(2,2,figsize=(8,8))
        axs = ax.flatten()

        axs[0].plot(x.T,y.T,'-',color='tab:blue')
        axs[0].plot(cpts[:,0],cpts[:,1],'.',color='tab:orange')
        axs[0].set_xlabel('X ('+xunit+')')
        axs[0].set_ylabel('Y ('+yunit+')')
        axs[0].set_aspect('equal')
        axs[0].set_xlim(-radius_air*1.1,radius_air*1.1)
        axs[0].set_ylim(-radius_air*1.1,radius_air*1.1)
        axs[0].set_title('Zoom Out')

        axs[1].plot(x.T,y.T,'-',color='tab:blue')
        axs[1].plot(cpts[:,0],cpts[:,1],'.',color='tab:orange')
        axs[1].set_xlabel('X ('+xunit+')')
        axs[1].set_ylabel('Y ('+yunit+')')
        axs[1].set_aspect('auto')
        axs[1].set_xlim(-radius_disk*1.1,radius_disk*1.1)
        #axs[1].set_ylim(-radius_disk*1.1,radius_disk*1.1)
        axs[1].set_title('Zoom In: Cantilever')

        axs[2].plot(x.T,y.T,'-',color='tab:blue')
        axs[2].plot(cpts[:,0],cpts[:,1],'.',color='tab:orange')
        axs[2].set_xlabel('X ('+xunit+')')
        axs[2].set_ylabel('Y ('+yunit+')')
        axs[2].set_aspect('equal')
        axs[2].set_xlim(-(radius_tip*4+height_gap),(radius_tip*4+height_gap))
        axs[2].set_ylim(-(radius_tip*4+height_gap),(radius_tip*4+height_gap))
        axs[2].set_title('Zoom In: Tip')

        axs[3].plot(x.T,y.T,'-',color='tab:blue')
        axs[3].plot(cpts[:,0],cpts[:,1],'.',color='tab:orange')
        axs[3].set_xlabel('X ($\mu$m)')
        axs[3].set_ylabel('Y ($\mu$m)')
        axs[3].set_aspect('equal')
        axs[3].set_xlim(-height_water*10,height_water*10)
        axs[3].set_ylim(-height_water*10,height_water*10)
        axs[3].set_title('Zoom In: Water')

        plt.tight_layout()
        plt.show()

