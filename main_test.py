#usage: python main.py
#output: solution to the Poission-Boltzmann equation

import numpy as np
import numpy.matlib
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time

#Domain discretization and required variables (from Jin 2002 P167)
#The volume V is subdivided into a number tetrahedral elements and
#the surface is broken into a number of triangular elements
#(1) array n(i,e) of size 4xM
#The element numbers and node numbers can be related by a 4xM integer array n(i,e)
#where i=1,2,3,4, e=1,2,3,...,M, and M denotes the total number of volume elements
#The array n(i,e) stores the global node number of the i^th node of the e^th element
#(2) vector nd(i) of length N1
#To facilitate imposition of the Dirichlet boundary condition on surface S1
#the integer vector nd(i) stores the global node number of the nodes that reside on
#the surface S1, where i=1,2,3,...,N1, and N1 denotes the total number of such nodes
#(3) array ns(i,e) of size 3xMs - not defined yet
#For the treatment of the Neumann boundary condition on surface S2
#the 3xMs integer array ns(i,e) holds information related to the surface triangular
#elements and their associated nodes residing on the surface S2, where i=1,2,3,
#e=1,2,3,...,Ms,and Ms is the total number of surface triangular elements on S2
#The array ns(i,e) stores the global number of the i^th node of the s^th elements
#(4) Other data that are needed include the coordinates of each node, the value of
#PDE coefficients for each volume elements, the prescribed value of u for each node
#on S1, and the value of the Neumann boundary condition coefficients for each surface
#triangular elements on S2

#Useful links
#(1) Quadrature on tetrahedra
#https://www.cfd-online.com/Wiki/Code:_Quadrature_on_Tetrahedra
#(2) Quadpy
#https://github.com/nschloe/quadpy
#(3) Shape function
#https://www.iue.tuwien.ac.at/phd/orio/node48.html
#(4) Another document about reference tetrahedra
#https://people.sc.fsu.edu/~jburkardt/presentations/cg_lab_fem_basis_tetrahedron.pdf

def set_globvars(mode):
    global nodes,elements,faces,midpoints #mesh variables
    global mask_d,node_d,node_notd,face_s #node indicies on S1; surface elements on S2
    global cx,cy,cz,alphax,alphay,alphaz,a,f,g,q,s #PDE coefficients
    
    #below parameters should be specified in an input file in the future
    ec=1.602e-19 #elementary charge [C]
    perm0=8.85e-12 #vacuum permittivity [F/m]
    kB=1.381e-23 #Boltzmann's constant [J/K]
    T=298 #room/ambient temperature [K]
    
    if mode=='1':
        prefix='box'
    elif mode=='2':
        prefix='box_1d'
        
    nodefile=prefix+'.1.node'
    elefile=prefix+'.1.ele'
    edgefile=prefix+'.1.edge'
    facefile=prefix+'.1.face'
    neighfile=prefix+'.1.neigh'
    trnfile=prefix+'.trn'
    
    #load mesh
    nodes=np.genfromtxt(prefix+'.1.node',skip_header=1,skip_footer=1,usecols=(1,2,3))
    node_flg=np.genfromtxt(prefix+'.1.node',skip_header=1,skip_footer=1,usecols=5,dtype='int')
    
    elements=np.genfromtxt(prefix+'.1.ele',skip_header=1,usecols=(1,2,3,4),dtype='int')
    zones=np.genfromtxt(prefix+'.1.ele',skip_header=1,usecols=5,dtype='int')
    
    faces=np.genfromtxt(prefix+'.1.face',skip_header=1,usecols=(1,2,3),dtype='int')
    face_flg=np.genfromtxt(prefix+'.1.face',skip_header=1,usecols=4,dtype='int')
    
    #extract Dirichelet nodes and Neuman faces
    elements=elements-1 #indices start from zero
    faces=faces-1 #indices start from zero
    if mode=='1':
        mask_d=(node_flg==2)|(node_flg==1)
        node_d=np.where(mask_d)[0] #node indicies on Dirichlet boundary
        node_notd=np.where(~mask_d)[0] #node indices not on Dirichlet boundary    
        face_s=[] #face elements on Neumann boundary
    elif mode=='2':
        mask_d=(node_flg==1)|((node_flg==2)&(nodes[:,-1]<-99.99))
        node_d=np.where(mask_d)[0] #node indices on Dirichlet boundary
        node_notd=np.where(~mask_d)[0] #node indices not on Dirichlet boundary
        face_s=faces[face_flg==2,:] #face elements on Neumann boundary but part will be shielded by Dirichlet B.C.
    
    nd=len(node_d)
    ns=len(face_s)
    print('Total number of nodes',len(nodes))
    print('Total number of elements',len(elements))
    print('Number of nodes on Dirichlet boundary',nd)
    print('Number of elements on Neuman boundary',ns)
    print('')
    
    #compute middle point of each elements (efficiency to be improved)
    nelem=len(elements[:,0])
    midpoints=np.zeros((nelem,3))
    for i in range(0,nelem):
        for j in range(0,3):
            midpoints[i,j]=.25*sum(nodes[elements[i,:],j])
    
    if mode=='1':
        #setup PDE variables for gravity modeling
        cx=np.ones(nelem)
        cy=np.ones(nelem)
        cz=np.ones(nelem)

        alphax=np.zeros(nelem)
        alphay=np.zeros(nelem)
        alphaz=np.zeros(nelem)

        a=np.zeros(nelem)
        f=np.zeros(nelem)
        f[zones==2]=0.1*1e8*6.67e-11*4*np.pi
        
        g=np.zeros(ns) #length of ns
        q=np.zeros(ns) #length of ns
        s=np.zeros(nd) #length of nd
    
    elif mode=='2':
        #setup PDE variable for GC equation
        cx=perm0*78.5*np.zeros(nelem)
        cy=perm0*78.5*np.zeros(nelem)
        cz=perm0*78.5*np.zeros(nelem)

        alphax=np.zeros(nelem)
        alphay=np.zeros(nelem)
        alphaz=np.zeros(nelem)

        #first iteration
        a=2*0.01*ec*ec/kB/T*np.ones(nelem)
        f=np.zeros(nelem) #will be reset outside

        g=np.zeros(ns) #length of ns
        q=np.zeros(ns) #length of ns
        s=np.zeros(nd) #length of nd
        
        #reset Dirichlet boundary condition on solid interface
        indices=np.where(node_flg==1)[0]
        s[:len(indices)]=50e-3
    
    return

def basis(xn,yn,zn,xr,yr,zr):
    Je=np.zeros((4,4))
    
    Je[0,:]=1
    Je[1,:]=xn #x-coordinates of four nodes in an element
    Je[2,:]=yn
    Je[3,:]=zn
    invJe=np.linalg.inv(Je)
    Ne=invJe.dot([1,xr,yr,zr])
    
    return Ne

def build_f(u):
    #input u on nodes output f in elements
    ec=1.602e-19 #elementary charge [C]
    perm0=8.85e-12 #vacuum permittivity [F/m]
    kB=1.381e-23 #Boltzmann's constant [J/K]
    T=298 #ambient temperature [K]
    f_nodes=-2*0.01*ec*(np.sinh(ec*u/kB/T)-ec*u/kB/T)
    
    nelem=len(elements)
    f=np.zeros(nelem)
    for i in range(nelem):
        sctr=elements[i,:4]
        xn=nodes[sctr,0]
        yn=nodes[sctr,1]
        zn=nodes[sctr,2]

        xr=midpoints[i,0]
        yr=midpoints[i,1]
        zr=midpoints[i,2]
        
        Ne=basis(xn,yn,zn,xr,yr,zr)
        f[i]=np.sum(Ne*f_nodes[sctr])
    
    return f

def build_b(nodes,elements,faces,cx,cy,cz,a,f,g,q,s):
    nnode=len(nodes)
    nelem=len(elements)
    nd=len(node_d)
    nnotd=len(node_notd)
    ns=len(face_s)

    print('Assembling the system of equations')
    start=time.time()
    b=np.zeros(nnode)
    for i in range(nelem):
        sctr=elements[i,:4]
        xn=nodes[sctr,0]
        yn=nodes[sctr,1]
        zn=nodes[sctr,2]
        junk,bv=build_quadvol(xn,yn,zn,cx[i],cy[i],cz[i],a[i],f[i])
        b[sctr]=b[sctr]+bv
    
    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')
    
    print('Incoorprating the Neumann boundary condition')
    start=time.time()
    for i in range(ns):
        sctr=face_s[i,:3]
        xn=nodes[sctr,0]
        yn=nodes[sctr,1]
        zn=nodes[sctr,2]
        junk,bs=build_quadsurf(xn,yn,zn,g[i],q[i])
        b[sctr]=b[sctr]+bs
    
    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')

    start=time.time()
    print('Incoorprating the Dirichlet boundary condition')
    b[node_d]=s
    for i in range(nnotd):
        sctr=node_notd[i]
        b[sctr]=b[sctr]-b_drop[i]
    
    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')
    
    return b


def build_system(nodes,elements,faces,cx,cy,cz,a,f,g,q,s):
    global b_drop
    
    nnode=len(nodes)
    nelem=len(elements)
    nd=len(node_d)
    nnotd=len(node_notd)
    ns=len(face_s)
    
    nt=nelem*16+ns*9+nnotd*nd*2
    I=np.zeros(nt)
    J=np.zeros(nt)
    X=np.zeros(nt)
    nt=0
    b=np.zeros(nnode)
    
    print('Assembling the system of equations')
    start=time.time()
    for i in range(nelem):
        sctr=elements[i,:4]
        xn=nodes[sctr,0]
        yn=nodes[sctr,1]
        zn=nodes[sctr,2]
        Kv,bv=build_quadvol(xn,yn,zn,cx[i],cy[i],cz[i],a[i],f[i])
        nt=nt+16
        
        #I holds the global row indices of Kv_ij, e.g. [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
        #J holds the global col indices of Kv_ij, e.g. [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        #X holds the entries of Kv_ij
        I[nt-16:nt]=np.matlib.repmat(sctr,4,1).flatten(order='F')
        J[nt-16:nt]=np.matlib.repmat(sctr,4,1).flatten(order='C')
        X[nt-16:nt]=Kv.flatten(order='C') #This oder is not important because Kv_ij=Kv_ji
        b[sctr]=b[sctr]+bv
    
    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')
    
    print('Incoorprating the Neumann boundary condition')
    start=time.time()
    for i in range(ns):
        sctr=face_s[i,:3]
        xn=nodes[sctr,0]
        yn=nodes[sctr,1]
        zn=nodes[sctr,2]
        Ks,bs=build_quadsurf(xn,yn,zn,g[i],q[i])
        nt=nt+9
        
        #I holds the global row indices of Ks_ij, e.g. [[1,1,1],[2,2,2],[3,3,3]]
        #J holds the global col indices of Ks_ij, e.g. [[1,2,3],[1,2,3],[1,2,3]]
        #X holds the entries of Ks_ij
        I[nt-9:nt]=np.matlib.repmat(sctr,3,1).flatten(order='F')
        J[nt-9:nt]=np.matlib.repmat(sctr,3,1).flatten(order='C')
        X[nt-9:nt]=Ks.flatten(order='C') #This order is not important because Ks_ij=Ks_ji
        b[sctr]=b[sctr]+bs
    
    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')

    #build matrix K using I,J,X
    #remember to manipulate elements in K using boolean index
    K=csr_matrix((X,(I,J)),shape=(nnode,nnode))
    
    start=time.time()
    print('Incoorprating the Dirichlet boundary condition')
    b[node_d]=s
    K[mask_d,:]=0
    K[mask_d,mask_d]=1
    K.eliminate_zeros()

    b_drop=np.zeros(nnotd)
    for i in range(nnotd):
        sctr=node_notd[i]
        #b[sctr]=b[sctr]-K[sctr,mask_d].dot(s)
        b_drop[i]=K[sctr,mask_d].dot(s)
        b[sctr]=b[sctr]-b_drop[i]
        
        #K[node_notd,node_d]=0
        K[sctr,mask_d]=0
    K.eliminate_zeros()

    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')
    return K,b
                

def build_quadvol(xn,yn,zn,cx,cy,cz,a,f):
    #modify this to accomadate matrix/vector PDE coefficients
    Kv=np.zeros((4,4))
    bv=np.zeros(4)
    Je=np.zeros((4,4))
    
    Je[0,:]=1
    Je[1,:]=xn #x-coordinates of four nodes in an element
    Je[2,:]=yn
    Je[3,:]=zn
    invJe=np.linalg.inv(Je)
    detJe=np.linalg.det(Je)
    vol=detJe/6
    
    for i in range(4):
        Kv[i,:]=(cx*invJe[i,1]*invJe[:,1]+cy*invJe[i,2]*invJe[:,2]+cz*invJe[i,3]*invJe[:,3]+a/20)*vol
        Kv[i,i]=Kv[i,i]+a*vol/20
        bv[i]=f*vol/4
    return Kv,bv
    
def build_quadsurf(xs,ys,zs,g,q):
    #modify this to accomodate matrix/vector PDE coefficients
    Ks=np.zeros((3,3))
    bs=np.zeros(3)
    
    a=[xs[1]-xs[0],ys[1]-ys[0],zs[1]-zs[0]]
    b=[xs[2]-xs[1],ys[2]-ys[1],zs[2]-zs[1]]
    c=np.cross(a,b)
    area=np.linalg.norm(c)/2
    
    for i in range(3):
        Ks[i,:]=q*area/12
        Ks[i,i]=Ks[i,i]+q*area/12
        bs[i]=g*area/3    
        
    return Ks,bs
    
if __name__=='__main__':
    #load mesh
    #build system equations
    #call solver
    #output results
    
    mode='2' #1D GC equation
    prefix='box_1d'

    set_globvars(mode)
    u=np.zeros(len(nodes))
    rec=nodes[40:59,:]
    output=np.zeros((19,5))
    output[:,0]=range(1,20)
    output[:,1:4]=rec
    output[:,4]=u[40:59]
    fname='box_1d_output%02d.txt'%(0)
    np.savetxt(fname,output,fmt='%d\t%10.2f\t%10.2f\t%10.2f\t%10.2e')

    f=build_f(u)
    [K,b]=build_system(nodes,elements,faces,cx,cy,cz,a,f,g,q,s)
    u=spsolve(K,b) #solver
    
    for i in range(1,21):
        print('iteration %d'%i)
        rec=nodes[40:59,:]
        output=np.zeros((19,5))
        output[:,0]=range(1,20)
        output[:,1:4]=rec
        output[:,4]=u[40:59]
        fname='box_1d_output%02d.txt'%(i)
        np.savetxt(fname,output,fmt='%d\t%10.2f\t%10.2f\t%10.2f\t%10.2e')
        
        f=build_f(u)
        b=build_b(nodes,elements,faces,cx,cy,cz,a,f,g,q,s)
        u=spsolve(K,b) #solver

#     if mode=='1':
#         rec=nodes[64:145,:]
#         output=np.zeros((81,5))
#         output[:,0]=range(1,82)
#         output[:,1:4]=rec
#         output[:,4]=u[64:145]
#         np.savetxt('output.txt',output,fmt='%d\t%10.2f\t%10.2f\t%10.2f\t%10.2e')
#     print(u)
    