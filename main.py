#usage: python main.py
#output: solution to the Poission-Boltzmann equation

import numpy as np
import numpy.matlib
from scipy.sparse import csr_matrix

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

def set_globvars():
    global echarge,perm0,kB,T #physical parameters
    global nodes,elements,faces,midpoints #mesh variables
    global nodes_d,nodes_nd,faces_s #nodes on S1; surface elements on S2
    global rel_perm,mobility,mobility_s #material variables
    global cplus_inf,cminus_inf,sigma_s #boundary conditions
    global cx,cy,cz,alphax,alphay,alphaz,a,f,g,q,s #PDE coefficients
    
    echarge=1.602e-19 #elementary charge [C]
    perm0=8.85e-12 #vacuum permittivity [F/m]
    kB=1.381e-23 #Boltzmann's constant [J/K]
    T=293 #room temperature [K]
    
    prefix='sphere'
    nodefile=prefix+'.1.node'
    elefile=prefix+'.1.ele'
    edgefile=prefix+'.1.edge'
    facefile=prefix+'.1.face'
    neighfile=prefix+'.1.neigh'
    trnfile=prefix+'.trn'
    
    #load nodes, elements, etc.
    nodes=np.genfromtxt(prefix+'.1.node',skip_header=1,skip_footer=1,usecols=(1,2,3))
    flags=np.genfromtxt(prefix+'.1.node',skip_header=1,skip_footer=1,usecols=5,dtype='int')
    
    elements=np.genfromtxt(prefix+'.1.ele',skip_header=1,usecols=(1,2,3,4),dtype='int')
    zones=np.genfromtxt(prefix+'.1.ele',skip_header=1,usecols=5,dtype='int')
    
    faces=np.genfromtxt(prefix+'.1.face',skip_header=1,usecols=(1,2,3),dtype='int')
    markers=np.genfromtxt(prefix+'.1.face',skip_header=1,usecols=4,dtype='int')
    
    nodes_d=np.where(flags==2)[0]+1 #nodes on Dirichlet boundary
    nodes_nd=np.where(flags!=2)[0]+1 #nodes not on Dirichlet boundary
    nd=len(nodes_d)
    
    faces_s=faces[markers==2,:] #elements on Neumann boundary
    ns=len(faces_s)
    
    #compute middle point of each elements (efficiency to be improved)
    nele=len(elements[:,0])
    midpoints=np.zeros((nele,3))
    for i in range(0,nele):
        for j in range(0,3):
            midpoints[i,j]=.25*sum(nodes[elements[i,:]-1,j])
    
    #set up relative permittivity [unitless] (individual function needed for complex model)
    rel_perm=np.zeros(nele)
    mask=zones==1
    rel_perm[mask]=4.5 #relative permittivity of solid
    mask=zones==2
    rel_perm[mask]=80 #relative permittivity of electrolyte
    
    #set up ion mobility [m^2/(Vs)] (individual function needed for complex model)
    mobility=np.zeros(nele) #ion mobility in [m^2/(Vs)]
    mask=zones==2
    mobility[mask]=5e-8 #ion mobility in electrolyte
    
    #set up ion mobility in Stern layer
    #mobility_s=np.zeros(ns)
    
    #set up boundary conditions
    cplus_inf=1 #ion concentration [mol/m^2] in the bulk electrolyte at infinity
    cminus_inf=1 #ion concentration [mol/m^2] in the bulk electrolyte at ininity
    sigma_s=-0.001 #surface charge density in [C/m^3 ?]
    
    #set up PDE variables for static case
    cx=perm0*rel_perm #permitivity [F/m] of length nele
    cy=perm0*rel_perm #permitivity [F/m] of length nele
    cz=perm0*rel_perm #permitivity [F/m] of length nele
    
    alphax=np.zeros(nele)
    alphay=np.zeros(nele)
    alphaz=np.zeros(nele)
    
    a=np.zeros(nele)
    f=-(cplus_inf+cminus_inf)*echarge*echarge/kB/T*np.ones(nele) #to be modified after linearization
    
    g=np.zeros(ns) #length of ns
    q=np.zeros(ns) #length of ns
    s=np.zeros(nd) #length of nd
    
    return

def build_system(nodes,elements,faces,cx,cy,cz,a,f,g,q,s):
    nnode=len(nodes)
    nele=len(elements)
    nd=len(nodes_d)
    nnd=len(nodes_nd)
    ns=len(faces_s)
    
    nt=nele*16+ns*9
    I=np.zeros(nt)
    J=np.zeros(nt)
    X=np.zeros(nt)
    nt=0
    b=np.zeros(nnode)
    
    print('Assembling the system of equations')
    for i in range(nele):
        sctr=elements[i,:4]-1
        xn=nodes[sctr,0]
        yn=nodes[sctr,1]
        zn=nodes[sctr,2]
        Kv,bv=build_quadvol(xn,yn,zn,cx[i],cy[i],cz[i],a[i],f[i])
        
#         print('Kv',Kv)
#         print('bv',bv)
#         print('sctr',sctr)
#         print('Row Index',np.matlib.repmat(sctr,4,1).flatten(order='F'))
#         print('Col Index',np.matlib.repmat(sctr,4,1).flatten(order='C'))
        
        nt=nt+16
        
        #I holds the global row indices of Kv_ij, e.g. [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
        #J holds the global col indices of Kv_ij, e.g. [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        #X holds the entries of Kv_ij
        I[nt-16:nt]=np.matlib.repmat(sctr,4,1).flatten(order='F')
        J[nt-16:nt]=np.matlib.repmat(sctr,4,1).flatten(order='C')
        X[nt-16:nt]=Kv.flatten(order='C') #This oder is not important because Kv_ij=Kv_ji
        b[sctr]=b[sctr]+bv
    
    print('Incorporation of the Neumann boundary condition')
    for i in range(ns):
        sctr=faces_s[i,:3]-1
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
    
    #build matrix K using I,J,X
    K=csr_matrix((X,(I,J)),shape=(nnode, nnode))    
    
    print('Incorporation of the Dirichlet boundary condition')
    b[nodes_d]=s
    for i in range(nnd):
        #if i is on the Dirichlet boundary, skip
        sctr=nodes_nd[i]-1
#         print(K[sctr,nodes_d].shape,np.expand_dims(s,axis=1).shape)
#         print(K[sctr,nodes_d]*np.expand_dims(s,axis=1))
#         break
        
        b[sctr]=b[sctr]-K[sctr,nodes_d].dot(s)
        K[sctr,nodes_d]=0
        K[nodes_d,sctr]=0
    
    print('K',K)
    print('b',b)
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
#     print('Je',Je)
#     print('Volume',vol)
    
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
    

# def build_qauadvol(nodes,lgpts,lgwts):
#     #modify this to accomodate matrix/vector PDE coefficients
#     Kv=np.zeros((4,4))
#     bv=np.zeros(4)
#     Je=np.zeros((3,3))
#     Je[:,0]=nodes[1,:]-nodes[0,:]
#     Je[:,1]=nodes[2,:]-nodes[0,:]
#     Je[:,2]=nodes[3,:]-nodes[0,:]
#     invJe=np.linalg.inv(Je)
#     detJe=np.linalg.det(Je)
#     dv=np.linalg.det([[1,1,1,1],nodes[:,0].T,nodes[:,1].T,nodes[:,2].T])/6
#     print(detJe/6,dv)
#     for i in range(len(lgwts)):
#         Ne=basis(lgpts[i,0],lgpts[i,1],lgpts[i,2])
#         Ge=grad(lgpts[i,0],lgpts[i,1],lgpts[i,2])
#         B=np.hstack(Ge*invJe,Ne)
#         Kv=Kv+B*np.diag([cx,cy,cz,a])*B.T*lgwt[i]*dv
#         bv=bv+Ne*f*lgwt[i]*dv
        
#     return Kv,bv

# def basis(xlocal,ylocal,zlocal):
#     #xlocal,ylocal,zlocal represent point in transformed coordinate system
#     #a=[0,1,0,0] #epsilon or xprime in transformed coordinate
#     #b=[0,0,1,0] #eta or yprime in transformed coordinate
#     #c=[0,0,1,1] #zeta or zprime in transformed coordinate
#     Ne=np.zeros(4)
#     Ne[0]=1-xlocal-ylocal-zlocal
#     Ne[1]=xlocal
#     Ne[2]=ylocal
#     Ne[3]=zlocal
#     return Ne

# def grad(xlocal,ylocal,zlocal):
#     #xlocal,ylocal,zlocal represent point in transformed coordinate system
#     #a=[0,1,0,0] #epsilon or xprime in transformed coordinate
#     #b=[0,0,1,0] #eta or yprime in transformed coordinate
#     #c=[0,0,1,1] #zeta or zprime in transformed coordinate    
#     Ge=np.zeros((4,3))
#     Ge[0,:]=[-1,-1,-1]
#     Ge[1,:]=[1,0,0]
#     Ge[2,:]=[0,1,0]
#     Ge[3,:]=[0,0,1]
#     return Ge

# def lgwt(N,a,b): #need to make sure b>a
#     #Computes the 1D sample points and weights for
#     #Gauss-Legendre quadrature within interval [-1,1]
#     x,w=np.polynomial.legendre.leggauss(N)
#     #Linear map from[-1,1] to [a,b]
#     x=(a*(1-x)+b*(1+x))/2
#     w=w*(b-a)/2
#     return x,w

if __name__=='__main__':
    #load mesh
    #build system equations
    #call solver
    #output results
    
    set_globvars()
    build_system(nodes,elements,faces,cx,cy,cz,a,f,g,q,s)
        
    #print(elements.shape)
    #print(np.polynomial.legendre.leggauss(5))
    #print(lgwt(5,-1,1))
#     np.random.seed(0)
#     x=np.random.rand(4)
#     y=np.random.rand(4)
#     z=np.random.rand(4)
#     #print(x[0],y[0],z[0])
#     print(basis(0,1,0))
#     print(grad(0,1,0))
    #build_qauadvol(nodes[1000,:])
    #scheme=quadpy.t3.schemes['shunn_ham_4']()
    #val=scheme.points
    #scheme.show()
    #print(scheme.points)
    #print(scheme.weights)
    #scheme=quadpy.t3.get_good_scheme(2)
#     scheme=quadpy.t3.schemes['shunn_ham_4']()
#     print(scheme)
#     print(scheme.points)
#     print(scheme.weights)
    #print(scheme)
    #print(dir(quadpy.t3))
    