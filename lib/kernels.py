import matplotlib.pyplot as plt
import matplotlib.tri as tri
plt.style.use('seaborn-poster')

import copy,datetime,os,subprocess,time
import numpy as np
import numpy.matlib
from scipy import constants
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

#set physicsical constants
E_C = constants.value(u'elementary charge') #1.602e-19 [C]
EPSILON_0 = constants.value(u'vacuum electric permittivity') #8.85e-12 [F/m]
N_A = constants.value(u'Avogadro constant') #6.022e23 [1/mol]
K_B = constants.value(u'Boltzmann constant') #1.381e-23 [J/K]
K_F = constants.value(u'Faraday constant') #96485.0 [C/mol]

#hard coded mesh indexing constants
ELEM_FLAG_SOLID = 1
ELEM_FLAG_WATER = 2
ELEM_FLAG_AIR = 3

EDGE_FLAG_SW_INTERFACE = 1
EDGE_FLAG_AW_INTERFACE = 2
EDGE_FLAG_EQUIPOTENTIAL_SURF = 3
EDGE_FLAG_AXIS_SYMMETRY = 4

EDGE_FLAG_TOP_BOUND = 11
EDGE_FLAG_BOTTOM_BOUND = 12
EDGE_FLAG_LEFT_BOUND = 13
EDGE_FLAG_RIGHT_BOUND = 14

def check_system():
    print('Run system check')
    print('Default numpy.int is %d bits'%np.iinfo(int).bits)
    print('Default numpy.float is %d bits'%np.finfo(float).bits)
    print('')
    return

def print_tstamp():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print('')
    return

def set_physics(c_ion,z_ion,mu_a,mu_s,rel_perm_a,rel_perm_i,
                sigma_solid,temperature,e_0,f_0,s_0,
                radius_a,is_metal):
    #set physical properties for SIP modeling
    #input c_ion: [float]*n_ion
    #input z_ion: [float]*n_ion
    #input mu_a: [float]*n_ion
    #input mu_s: float
    #input rel_perm_a: float
    #input rel_perm_i: float
    #input sigma_solid: float
    #input temperature: float
    #input e_0: [nfloat]*2 for [E_x, E_y] at infinity
    #input f_0: nested list, [[x_coord, y_coord, charge_density]*3 for i in range(n_source)]
    #input s_0: float, voltage on equipotential surface
    
    #compute derived variables
    C_ion = [val*N_A for val in c_ion]
    Q_ion = [val*E_C for val in z_ion]
    Diff_a = [val*K_B*temperature/E_C for val in mu_a]
    Diff_s = mu_s*K_B*temperature/E_C
    perm_a = rel_perm_a*EPSILON_0
    perm_i = rel_perm_i*EPSILON_0
    
    n_ion = len(c_ion)
    lambda_d = [0.0]*n_ion
    for i in range(n_ion):
        lambda_d[i] = np.sqrt(perm_a*K_B*temperature/2/Q_ion[i]/Q_ion[i]/C_ion[i])
        
    #output physics
    physics = {}
    physics['c_ion'] = c_ion
    physics['z_ion'] = z_ion
    physics['mu_a'] = mu_a
    physics['mu_s'] = mu_s
    physics['perm_a'] = perm_a
    physics['perm_i'] = perm_i
    physics['C_ion'] = C_ion
    physics['Q_ion'] = Q_ion
    physics['Diff_a'] = Diff_a
    physics['Diff_s'] = Diff_s
    physics['sigma_solid'] = sigma_solid
    physics['temperature'] = temperature
    physics['debye_length'] = lambda_d
    
    physics['e_0'] = e_0
    physics['f_0'] = f_0
    physics['s_0'] = s_0
    
    physics['radius_a'] = radius_a
    physics['is_metal'] = is_metal
    
    return physics

def set_survey(ratio,freq):
    #input ratio: list or ndarray of float
    #input freq: list of ndarray of float
    survey = {}
    survey['ratio'] = np.array(ratio,dtype=float)
    survey['freq'] = np.array(freq,dtype=float)
    
    return survey

def set_mesh(mesh_prefix,dist_factor):
    #load mesh (stability needs to be improved)
    print('Reading %s.1.node'%mesh_prefix)
    nodes = np.genfromtxt(mesh_prefix+'.1.node',skip_header=1,skip_footer=0,usecols=(1,2),dtype=float)
    node_flags = np.genfromtxt(mesh_prefix+'.1.node',skip_header=1,skip_footer=0,usecols=3,dtype=int)

    print('Reading %s.1.ele'%mesh_prefix)
    elements = np.genfromtxt(mesh_prefix+'.1.ele',skip_header=1,usecols=(1,2,3),dtype=int)
    elem_flags = np.genfromtxt(mesh_prefix+'.1.ele',skip_header=1,usecols=4,dtype=int)

    print('Reading %s.1.edge'%mesh_prefix)
    edges = np.genfromtxt(mesh_prefix+'.1.edge',skip_header=1,usecols=(1,2),dtype=int)
    edge_flags = np.genfromtxt(mesh_prefix+'.1.edge',skip_header=1,usecols=3,dtype=int)

    #adjust indices to start from zero
    elements = elements-1
    edges = edges-1
    
    #scale nodes from meter to micro-meter
    nodes = nodes*1e-6*dist_factor
    
    n_node = len(nodes)
    n_elem = len(elements)
    n_edge = len(edges)
    print('THE NUMBER OF NODES IS: %d'%n_node)
    print('THE NUMBER OF ELEMENTS IS: %d'%n_elem)
    print('THE NUMBER OF EDGES IS: %d'%n_edge)
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #determine certain elements and edges indices
    in_air = elem_flags==ELEM_FLAG_AIR
    in_water = elem_flags==ELEM_FLAG_WATER
    in_solid = elem_flags==ELEM_FLAG_SOLID
    
    with_stern = edge_flags==EDGE_FLAG_SW_INTERFACE
    with_equipotential = edge_flags==EDGE_FLAG_EQUIPOTENTIAL_SURF
    with_axis_symmetry = edge_flags==EDGE_FLAG_AXIS_SYMMETRY
    
    with_top_bound = edge_flags==EDGE_FLAG_TOP_BOUND
    with_bottom_bound = edge_flags==EDGE_FLAG_BOTTOM_BOUND
    with_left_bound = edge_flags==EDGE_FLAG_LEFT_BOUND
    with_right_bound = edge_flags==EDGE_FLAG_RIGHT_BOUND
        
    #--------------------------------------------------------------------------
    #assign current processor certain elements and edges to compute
    elem_proc = np.zeros(n_elem,dtype=bool)
    edge_proc = np.zeros(n_edge,dtype=bool)
    
    elem_proc = elem_proc|((in_air|in_water)|in_solid)
    edege_proc = edge_proc|with_stern
    
    #--------------------------------------------------------------------------
    #compute middle points of elements and edges
    elem_mids = np.zeros((n_elem,2),dtype=float)
    x_node = nodes[elements,0] #(n_elem,3)
    y_node = nodes[elements,1] #(n_elem,3)
    elem_mids[:,0] = np.sum(x_node,axis=1)/3.0
    elem_mids[:,1] = np.sum(y_node,axis=1)/3.0
    
    edge_mids = np.zeros((n_edge,2),dtype=float)
    x_node = nodes[edges,0] #(n_edge,2)
    y_node = nodes[edges,1] #(n_edge,2)
    edge_mids[:,0] = np.sum(x_node,axis=1)/2.0
    edge_mids[:,1] = np.sum(y_node,axis=1)/2.0
    
    #--------------------------------------------------------------------------
    #determine the axis of symmetry
    x = nodes[edges[with_axis_symmetry,:],0].flatten(order='C')
    y = nodes[edges[with_axis_symmetry,:],1].flatten(order='C')
    if np.linalg.norm(x)==0 and len(x)>0:
        print('AXIS OF SYMMETRY: ALONG Y')
        node_factor = nodes[:,0]
        elem_factor = elem_mids[:,0]
        edge_factor = edge_mids[:,0] #needs to be verified
        axis_symmetry_label = 'x'
    elif np.linalg.norm(y)==0 and len(y)>0:
        print('AXIS OF SYMMETRY: ALONG X')
        node_factor = nodes[:,1]
        elem_factor = elem_mids[:,1]
        edge_factor = edge_mids[:,1] #needs to be verified
        axis_symmetry_label = 'y'
    else:
        print('AXIS OF SYMMETRY: NONE')
        node_factor = np.ones(len(nodes))
        elem_factor = np.ones(len(elements))
        edge_factor = np.ones(len(edges))
        axis_symmetry_label = 'none'
        
    print('node_flags',np.unique(node_flags))
    print('elem_flags',np.unique(elem_flags))
    print('edge_flags',np.unique(edge_flags))
    print('')
    
    #compute shape functions and middle points
    elem_shape_fun,area = shape_fun_2d(nodes,elements,elem_proc)
    edge_shape_fun,length = shape_fun_1d(nodes,edges,edge_proc)
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #output variables
    mesh = {}
    mesh['nodes'] = nodes
    mesh['elements'] = elements
    mesh['edges'] = edges

    mesh['node_flags'] = node_flags
    mesh['elem_flags'] = elem_flags
    mesh['edge_flags'] = edge_flags
    
    mesh['node_factor'] = node_factor
    mesh['elem_factor'] = elem_factor
    mesh['edge_factor'] = edge_factor
        
    mesh['elem_shape_fun'] = elem_shape_fun
    mesh['elem_area'] = area
    mesh['elem_mids'] = elem_mids
    mesh['elem_proc'] = elem_proc
    
    mesh['edge_shape_fun'] = edge_shape_fun
    mesh['edge_len'] = length
    mesh['edge_mids'] = edge_mids
    mesh['edge_proc'] = edge_proc
    
    mesh['in_air'] = in_air
    mesh['in_water'] = in_water
    mesh['in_solid'] = in_solid
    
    mesh['with_stern'] = with_stern
    mesh['with_equipotential'] = with_equipotential
    mesh['with_axis_symmetry'] = with_axis_symmetry
    
    mesh['with_top_bound'] = with_top_bound
    mesh['with_bottom_bound'] = with_bottom_bound
    mesh['with_left_bound'] = with_left_bound
    mesh['with_right_bound'] = with_right_bound
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #display mesh
    disp_factor = 1e6/dist_factor
    x = nodes[:,0]*disp_factor
    y = nodes[:,1]*disp_factor
    
    fig,ax = plt.subplots(figsize=(8,8))
    if axis_symmetry_label=='x':
        ax.triplot(x,y,elements[:,:],linewidth=0.2,color='tab:blue')
        ax.triplot(-x,y,elements[:,:],linewidth=0.2,color='tab:blue',alpha=0.5)
    elif axis_symmetry_label=='y':
        ax.triplot(x,y,elements[:,:],linewidth=0.2,color='tab:blue')
        ax.triplot(x,-y,elements[:,:],linewidth=0.2,color='tab:blue',alpha=0.5)
    else:
        ax.triplot(x,y,elements[:,:],linewidth=0.2,color='tab:blue')
    ax.set_xlabel('X $(\mu m)$')
    ax.set_ylabel('Y $(\mu m)$')
    ax.set_aspect('equal')
    ax.set_title('Zoom Out')
    plt.show()
    
    return mesh

def set_domain(mesh,physics,dist_factor,run_mode):
    #local variables to be defined
    #conc_stat
    #pot_stat
    
    #initialize local variables
    n_dim = 2
    n_node = len(mesh['nodes'])
    n_elem = len(mesh['elements'])
    n_edge = len(mesh['edges'])
    n_ion = len(physics['c_ion'])
    
    ind_c = np.arange(0,n_ion)
    ind_p = np.arange(n_ion,n_ion+1)
    ind_s = np.arange(n_ion+1,n_ion+2)
    freq = 1.0 #placeholder
    
    #shortcut to varibles in physics
    c_ion = physics['c_ion']
    z_ion = physics['z_ion']
    Q_ion = physics['Q_ion']
    mu_a = physics['mu_a']
    Diff_a = physics['Diff_a']    
    perm_a = physics['perm_a']
    perm_i = physics['perm_i']
    temperature = physics['temperature']
    f_0 = physics['f_0']
    
    #shortcut to variables in mesh
    nodes = mesh['nodes']
    elements = mesh['elements']
    edges = mesh['edges']
    
    node_flags = mesh['node_flags']
    elem_flags = mesh['elem_flags']
    edge_flags = mesh['edge_flags']
    
    in_air = mesh['in_air']
    in_water = mesh['in_water']
    in_solid = mesh['in_solid']
    
    node_factor = mesh['node_factor']
    elem_factor = mesh['elem_factor']
    edge_factor = mesh['edge_factor']
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #initialize domain keys and values
    domain = {}
    domain['inside'] = (in_air|in_water)|in_solid
    domain['c_x']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['c_y']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['alpha_x']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['alpha_y']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['beta_x']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['beta_y']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['gamma_x']=np.zeros((n_elem,n_ion+2),dtype=float)
    domain['gamma_y']=np.zeros((n_elem,n_ion+2),dtype=float)
    domain['a']=np.zeros((n_elem,n_ion+2,n_ion+2),dtype=float)
    domain['f']=np.zeros((n_elem,n_ion+2),dtype=float)
    domain['a_n']=np.zeros((n_node,n_ion+2,n_ion+2),dtype=float) #test purpose
    domain['f_n']=np.zeros((n_node,n_ion+2),dtype=float) #test purpose
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set domain properties in the air
    if run_mode==0:
        for i in range(n_ion):
            domain['c_x'][in_air,i,i] = 0.0
        domain['c_x'][in_air,ind_p,ind_p] = EPSILON_0
        domain['c_x'][in_air,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['c_y'][in_air,i,i] = 0.0
        domain['c_y'][in_air,ind_p,ind_p] = EPSILON_0
        domain['c_y'][in_air,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['a'][in_air,i,i] = 0.0
        domain['a'][in_air,ind_p,ind_p] = 0.0
        domain['a'][in_air,ind_s,ind_s] = 0.0
    else:
        for i in range(n_ion):
            domain['c_x'][in_air,i,i] = 0.0
        domain['c_x'][in_air,ind_p,ind_p] = EPSILON_0
        domain['c_x'][in_air,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['c_y'][in_air,i,i] = 0.0
        domain['c_y'][in_air,ind_p,ind_p] = EPSILON_0
        domain['c_y'][in_air,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['a'][in_air,i,i] = 0.0
        domain['a'][in_air,ind_p,ind_p] = 0.0
        domain['a'][in_air,ind_s,ind_s] = 0.0
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set domain properties in the water
    if run_mode==0:
        for i in range(n_ion):
            domain['c_x'][in_water,i,i] = 0.0
        domain['c_x'][in_water,ind_p,ind_p] = perm_a
        domain['c_x'][in_water,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['c_y'][in_water,i,i] = 0.0
        domain['c_y'][in_water,ind_p,ind_p] = perm_a
        domain['c_y'][in_water,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['a'][in_water,i,i] = 0.0
        domain['a'][in_water,ind_p,ind_p] = 0.0
        domain['a'][in_water,ind_s,ind_s] = 0.0
    else:
        for i in range(n_ion):
            conc_stat = c_ion[i]*np.exp(-Q_ion[i]*pot_stat[:,0]/K_B/temperature)
            domain['c_x'][in_water,i,i] = Diff_a[i]
            domain['c_x'][in_water,i,ind_p] = mu_a[i]*z_ion[i]*conc_stat[in_water]
        domain['c_x'][in_water,ind_p,ind_p] = perm_a
        domain['c_x'][in_water,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            conc_stat = c_ion[i]*np.exp(-Q_ion[i]*pot_stat[:,0]/K_B/temperature)
            domain['c_y'][in_water,i,i] = Diff_a[i]
            domain['c_y'][in_water,i,ind_p] = mu_a[i]*z_ion[i]*conc_stat[in_water]
        domain['c_y'][in_water,ind_p,ind_p] = perm_a
        domain['c_y'][in_water,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['alpha_x'][in_water,i,i] = mu_a[i]*z_ion[i]*pot_stat[in_water,1]

        for i in range(n_ion):
            domain['alpha_y'][in_water,i,i] = mu_a[i]*z_ion[i]*pot_stat[in_water,2]

        for i in range(n_ion):
            domain['a'][in_water,i,i] = 1.0*freq
            domain['a'][in_water,ind_p,i] = -z_ion[i]*K_F
        domain['a'][in_water,ind_p,ind_p] = 0.0
        domain['a'][in_water,ind_s,ind_s] = 0.0

    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set domain properties in the solid
    if run_mode==0:
        for i in range(n_ion):
            domain['c_x'][in_solid,i,i] = 0.0
        domain['c_x'][in_solid,ind_p,ind_p] = perm_i
        domain['c_x'][in_solid,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['c_y'][in_solid,i,i] = 0.0
        domain['c_y'][in_solid,ind_p,ind_p] = perm_i
        domain['c_y'][in_solid,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['a'][in_solid,i,i] = 0.0
        domain['a'][in_solid,ind_p,ind_p] = 0.0
        domain['a'][in_solid,ind_s,ind_s] = 0.0
    else:
        for i in range(n_ion):
            domain['c_x'][in_solid,i,i] = 0.0
        domain['c_x'][in_solid,ind_p,ind_p] = perm_i
        domain['c_x'][in_solid,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['c_y'][in_solid,i,i] = 0.0
        domain['c_y'][in_solid,ind_p,ind_p] = perm_i
        domain['c_y'][in_solid,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            domain['a'][in_solid,i,i] = 0.0
        domain['a'][in_solid,ind_p,ind_p] = 0.0
        domain['a'][in_solid,ind_s,ind_s] = 0.0
        
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set point source for electric field
    for i in range(len(f_0)):
        x = f_0[i][0]
        y = f_0[i][1]
        ind_n = np.argmin((nodes[:,0]-x)**2+(nodes[:,1]-y)**2)
        domain['f_n'][ind_n,ind_p] = f_0[i][2]
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #multiply domain properties by elem_factor
    for i in range(n_ion+2):
        for j in range(n_ion+2):
            domain['c_x'][:,i,j] = domain['c_x'][:,i,j]*elem_factor
            domain['c_y'][:,i,j] = domain['c_y'][:,i,j]*elem_factor
            domain['alpha_x'][:,i,j] = domain['alpha_x'][:,i,j]*elem_factor
            domain['alpha_y'][:,i,j] = domain['alpha_y'][:,i,j]*elem_factor
            domain['beta_x'][:,i,j] = domain['beta_x'][:,i,j]*elem_factor
            domain['beta_y'][:,i,j] = domain['beta_y'][:,i,j]*elem_factor
            domain['a'][:,i,j] = domain['a'][:,i,j]*elem_factor
            domain['a_n'][:,i,j] = domain['a_n'][:,i,j]*node_factor
        
        domain['gamma_x'][:,i] = domain['gamma_x'][:,i]*elem_factor
        domain['gamma_y'][:,i] = domain['gamma_y'][:,i]*elem_factor
        domain['f'][:,i] = domain['f'][:,i]*elem_factor
        domain['f_n'][:,i] = domain['f_n'][:,i]*node_factor
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #scale domain properties over distance
    domain['c_x'][:] = domain['c_x']*dist_factor**2
    domain['c_y'][:] = domain['c_y']*dist_factor**2
    domain['alpha_x'][:] = domain['alpha_x']*dist_factor
    domain['alpha_y'][:] = domain['alpha_y']*dist_factor
    domain['beta_x'][:] = domain['beta_x']*dist_factor
    domain['beta_y'][:] = domain['beta_y']*dist_factor
    domain['a'][:] = domain['a']*1.0
    domain['a_n'][:] = domain['a_n']*1.0

    domain['gamma_x'][:] = domain['gamma_x']*dist_factor
    domain['gamma_y'][:] = domain['gamma_y']*dist_factor
    domain['f'][:] = domain['f']*1.0
    domain['f_n'][:] = domain['f_n']*1.0
    
    return domain

def set_stern(mesh,physics,dist_factor,run_mode):
    #initialize local variables
    n_dim = 2
    n_node = len(mesh['nodes'])
    n_elem = len(mesh['elements'])
    n_edge = len(mesh['edges'])
    n_ion = len(physics['c_ion'])
    
    ind_c = np.arange(0,n_ion)
    ind_p = np.arange(n_ion,n_ion+1)
    ind_s = np.arange(n_ion+1,n_ion+2)
    freq = 1.0 #placeholder
    sigma_stern = 1.0 #placeholder
    
    #shortcut to variables in physics
    mu_s = physics['mu_s']
    Diff_s = physics['Diff_s']
    is_metal = physics['is_metal']
    
    #shortcut to variables in mesh
    nodes = mesh['nodes']
    elements = mesh['elements']
    edges = mesh['edges']
    
    node_flags = mesh['node_flags']
    elem_flags = mesh['elem_flags']
    edge_flags = mesh['edge_flags']
        
    node_factor = mesh['node_factor']
    elem_factor = mesh['elem_factor']
    edge_factor = mesh['edge_factor']
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #initialize stern keys and values
    stern = {}
    stern['inside'] = np.zeros(n_edge,dtype=bool)
    stern['c_x'] = np.zeros((n_edge,n_ion+2,n_ion+2),dtype=float)
    stern['alpha_x'] = np.zeros((n_edge,n_ion+2,n_ion+2),dtype=float)
    stern['beta_x'] = np.zeros((n_edge,n_ion+2,n_ion+2),dtype=float)
    stern['gamma_x'] = np.zeros((n_edge,n_ion+2),dtype=float)
    stern['a'] = np.zeros((n_edge,n_ion+2,n_ion+2),dtype=float)
    stern['f'] = np.zeros((n_edge,n_ion+2),dtype=float)

    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set stern properties
    with_stern = mesh['with_stern']
    stern['inside'][:] = with_stern
    if run_mode==0:
        for i in range(n_ion):
            stern['c_x'][with_stern,i,i] = 0.0
        stern['c_x'][with_stern,ind_p,ind_p] = 0.0
        stern['c_x'][with_stern,ind_s,ind_s] = 0.0

        for i in range(n_ion):
            stern['a'][with_stern,i,i] = 0.0
        stern['a'][with_stern,ind_p,ind_p] = 0.0
        stern['a'][with_stern,ind_s,ind_s] = 0.0
    else:
        for i in range(n_ion):
            stern['c_x'][with_stern,i,i] = 0.0
        stern['c_x'][with_stern,ind_p,ind_p] = 0.0
        stern['c_x'][with_stern,ind_s,ind_p] = mu_s*sigma_stern*int(not is_metal)
        stern['c_x'][with_stern,ind_s,ind_s] = Diff_s*int(not is_metal)

        for i in range(n_ion):
            stern['a'][with_stern,i,i] = 0.0
        stern['a'][with_stern,ind_p,ind_p] = 0.0
        stern['a'][with_stern,ind_s,ind_s] = 1.0*freq*int(not is_metal)

    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #multiply stern properties by edge_factor
    for i in range(n_ion+2):
        for j in range(n_ion+2):
            stern['c_x'][:,i,j] = stern['c_x'][:,i,j]*edge_factor
            stern['alpha_x'][:,i,j] = stern['alpha_x'][:,i,j]*edge_factor
            stern['beta_x'][:,i,j] = stern['beta_x'][:,i,j]*edge_factor
            stern['a'][:,i,j] = stern['a'][:,i,j]*edge_factor
        
        stern['gamma_x'][:,i] = stern['gamma_x'][:,i]*edge_factor
        stern['f'][:,i] = stern['f'][:,i]*edge_factor
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #scale stern properties over distance
    stern['c_x'][:] = stern['c_x']*dist_factor**2
    stern['alpha_x'][:] = stern['alpha_x']*dist_factor
    stern['beta_x'][:] = stern['beta_x']*dist_factor
    stern['a'][:] = stern['a']*1.0

    stern['gamma_x'][:] = stern['gamma_x']*dist_factor
    stern['f'][:] = stern['f']*1.0
    
    return stern

def set_robin(mesh,physics,dist_factor,run_mode):
    #initialize local variables
    n_dim = 2
    n_node = len(mesh['nodes'])
    n_elem = len(mesh['elements'])
    n_edge = len(mesh['edges'])
    n_ion = len(physics['c_ion'])
    
    ind_c = np.arange(0,n_ion)
    ind_p = np.arange(n_ion,n_ion+1)
    ind_s = np.arange(n_ion+1,n_ion+2)
    freq = 1.0 #placeholder
    sigma_diffuse = 1.0 #placeholder
    
    #shortcut to variables in physics
    is_metal = physics['is_metal']
    
    #shortcut to variables in mesh
    nodes = mesh['nodes']
    elements = mesh['elements']
    edges = mesh['edges']
    
    node_flags = mesh['node_flags']
    elem_flags = mesh['elem_flags']
    edge_flags = mesh['edge_flags']
    
    node_factor = mesh['node_factor']
    elem_factor = mesh['elem_factor']
    edge_factor = mesh['edge_factor']
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #initialize robin keys and values
    robin = {}
    robin['with_3rd_kind_bc'] = np.zeros(n_edge,dtype=bool)
    robin['g_s'] = np.zeros((n_edge,n_ion+2),dtype=float)
    robin['q_s'] = np.zeros((n_edge,n_ion+2,n_ion+2),dtype=float)

    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set robin properties
    with_3rd_kind_bc = mesh['with_stern']
    robin['with_3rd_kind_bc'][:] = with_3rd_kind_bc
    if run_mode==0:
        robin['g_s'][with_3rd_kind_bc,ind_p] = sigma_diffuse*int(not is_metal)
        robin['q_s'][with_3rd_kind_bc,ind_p,ind_s] = 0.0
    else:
        robin['g_s'][with_3rd_kind_bc,:] = 0.0
        robin['q_s'][with_3rd_kind_bc,ind_p,ind_s] = -1.0*int(not is_metal)
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #multiply robin properties by edge_factor
    for i in range(n_ion+2):
        for j in range(n_ion+2):
            robin['q_s'][:,i,j] = robin['q_s'][:,i,j]*edge_factor
        
        robin['g_s'][:,i] = robin['g_s'][:,i]*edge_factor
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #scale robin properties over distance
    robin['q_s'][:] = robin['q_s']*dist_factor
    robin['g_s'][:] = robin['g_s']*dist_factor
    
    return robin

def set_dirichlet(mesh,physics,dist_factor,run_mode):
    #initialize local variables
    n_dim = 2
    n_node = len(mesh['nodes'])
    n_elem = len(mesh['elements'])
    n_edge = len(mesh['edges'])
    n_ion = len(physics['c_ion'])
    
    ind_c = np.arange(0,n_ion)
    ind_p = np.arange(n_ion,n_ion+1)
    ind_s = np.arange(n_ion+1,n_ion+2)
    freq = 1.0 #placeholder
    
    #shortcut to variables in physics
    e_0 = physics['e_0']
    s_0 = physics['s_0']
    is_metal = physics['is_metal']
    
    #shortcut to variables in mesh
    nodes = mesh['nodes']
    elements = mesh['elements']
    edges = mesh['edges']
    
    node_flags = mesh['node_flags']
    elem_flags = mesh['elem_flags']
    edge_flags = mesh['edge_flags']
    
    node_factor = mesh['node_factor']
    elem_factor = mesh['elem_factor']
    edge_factor = mesh['edge_factor']
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #initialize variables for the 1st kind b.c.
    dirichlet = {}
    dirichlet['on_1st_kind_bc'] = np.zeros((n_node,n_ion+2),dtype=bool)
    dirichlet['s_n'] = np.zeros((n_node,n_ion+2),dtype=float)
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #set material properties for the 1st kind b.c. at infinity
    with_outer_bound = mesh['with_top_bound']
    with_outer_bound = with_outer_bound|mesh['with_bottom_bound']
    with_outer_bound = with_outer_bound|mesh['with_left_bound']
    with_outer_bound = with_outer_bound|mesh['with_right_bound']
    ind_n = np.unique(edges[with_outer_bound,:].flatten(order='C'))
    for i in range(n_ion): #for i^th ion
        dirichlet['on_1st_kind_bc'][ind_n,i] = True
        dirichlet['s_n'][ind_n,i] = 0.0
    
    if run_mode==0:
        dirichlet['on_1st_kind_bc'][ind_n,ind_p] = True #for potential
        dirichlet['s_n'][ind_n,ind_p] = 0.0
    else:
        dirichlet['on_1st_kind_bc'][ind_n,ind_p] = True #for potential
        dirichlet['s_n'][ind_n,ind_p] = -nodes[ind_n,0]*e_0[0]-nodes[ind_n,1]*e_0[1]
#         print(-nodes[ind_n,0]*e_0[0]-nodes[ind_n,1]*e_0[1])
#         print(e_0[:])
    
    #set material properties for the 1st kind b.c. on equipotential surface
    with_inner_bound = mesh['with_equipotential']
    ind_n = np.unique(edges[with_inner_bound,:].flatten(order='C'))
    dirichlet['on_1st_kind_bc'][ind_n,ind_p] = True
    dirichlet['s_n'][ind_n,ind_p] = s_0
    
    #set zero potential and surface charge density
    #in Stern layer if the solid particle is metal
    with_inner_bound = mesh['with_stern']*is_metal
    ind_n = np.unique(edges[with_inner_bound,:].flatten(order='C'))
    dirichlet['on_1st_kind_bc'][ind_n,ind_p] = True
    dirichlet['s_n'][ind_n,ind_p] = 0.0
    dirichlet['on_1st_kind_bc'][ind_n,ind_s] = True
    dirichlet['s_n'][ind_n,ind_s] = 0.0
    
    #set zero ion concentrations and potential
    #inside the solid if the solid particle is metal
    #except for the nodes on the boundary of the solid
    on_selected_nodes = np.ones(n_node,dtype=bool)
    mask = mesh['in_air']|mesh['in_water']
    ind_n = np.unique(elements[mask,:].flatten(order='C'))
    on_selected_nodes[ind_n] = False
    on_selected_nodes[:] = on_selected_nodes*is_metal
    dirichlet['on_1st_kind_bc'][on_selected_nodes,:-1] = True
    dirichlet['s_n'][on_selected_nodes,:-1] = 0.0
    
    #impose zero ion concentrations outside water
    on_selected_nodes = np.ones(n_node,dtype=bool)
    mask = mesh['in_water']
    ind_n = np.unique(elements[mask,:].flatten(order='C'))
    on_selected_nodes[ind_n] = False
    dirichlet['on_1st_kind_bc'][on_selected_nodes,:-2] = True
    dirichlet['s_n'][on_selected_nodes,:-2] = 0.0
    
    #deactivate unused nodes in the air, water, and solid
    on_selected_nodes = np.ones(n_node,dtype=bool)
    mask = (mesh['in_air']|mesh['in_water'])|mesh['in_solid']
    ind_n = np.unique(elements[mask,:].flatten(order='C'))
    on_selected_nodes[ind_n] = False
    dirichlet['on_1st_kind_bc'][on_selected_nodes,:-1] = True
    dirichlet['s_n'][on_selected_nodes,:-1] = 0.0
    
    #deactivate unused nodes in Stern layer
    on_selected_nodes = np.ones(n_node,dtype=bool)
    mask = mesh['with_stern']
    ind_n = np.unique(edges[mask,:].flatten(order='C'))
    on_selected_nodes[ind_n] = False
    dirichlet['on_1st_kind_bc'][on_selected_nodes,-1] = True
    dirichlet['s_n'][on_selected_nodes,-1] = 0.0
    
    #set zero ion concentrations and surface charge density for run_mode 0
    if run_mode==0:
        for i in range(n_ion):
            dirichlet['on_1st_kind_bc'][:,i] = True
            dirichlet['s_n'][:,i] = 0.0
        
        dirichlet['on_1st_kind_bc'][:,ind_s] = True
        dirichlet['s_n'][:,ind_s] = 0.0
        
    return dirichlet

def shape_fun_2d(nodes,elements,elem_proc):
    #compute shape function Je for triangular elements
    #input: nodes.shape (n_node,2)
    #input: elements.shape (n_nelem,3)
    #output: Je.shape(n_elem,3,3)
    #output: area.shape (n_elem,)
    
    print('Computing shape functions of triangular elements')
    print('This will take a minute')
    start = time.time()
    
    n_elem = len(elements)
    Je = np.zeros((n_elem,3,3),dtype=float)
    area = np.zeros(n_elem,dtype=float)
    
    x_node = nodes[elements,0] #(n_elem,3)
    y_node = nodes[elements,1] #(n_elem,3)
    
    for i in range(n_elem):
        if elem_proc[i]:
            A = np.ones((3,3),dtype=float)
            A[1,:] = x_node[i,:]
            A[2,:] = y_node[i,:]
            Je[i,:,:] = np.linalg.inv(A)
            area[i] = np.abs(np.linalg.det(A)/2.0)
        else:
            continue
    
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return Je,area

def shape_fun_1d(nodes,edges,edge_proc):
    #compute shape function Je for line segments
    #need to convert 2D line to 1D line by
    #rotating 2D line along vector n to x
    
    #input: nodes.shape (n_node,2)
    #input: edges.shape (n_edge,2)
    #output: Je.shape(n_edge,2,2)
    #output: length.shape (n_edge,)
    
    print('Computing shape functions of line segments')
    print('This will take a minute')
    start = time.time()
    
    n_edge = len(edges)
    Je = np.zeros((n_edge,2,2),dtype=float)
    length = np.zeros(n_edge,dtype=float)
    
    x_node = nodes[edges,0] #(n_edge,2)
    y_node = nodes[edges,1] #(n_edge,2)
    
    for i in range(n_edge):
        if edge_proc[i]==True:
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

            #next compute Je and length
            B = np.ones((2,2),dtype=float)
            x_loc = np.zeros(2,dtype=float)
            x_loc[0] = R[0,0]*x_node[i,0]+R[0,1]*y_node[i,0]
            x_loc[1] = R[0,0]*x_node[i,1]+R[0,1]*y_node[i,1]
            B[1,:] = x_loc
            Je[i,:,:] = np.linalg.inv(B)
            length[i] = np.abs(x_loc[1]-x_loc[0])
        else:
            continue
    
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return Je,length

def rotate_lines(nodes,edges):
    #rotate 2D line along vector n to x
    #input: nodes.shape (n_node,2)
    #input: edges.shape (n_edge,2)
    #output: R.shape (n_edge,2,2)
    
    print('Computing rotation matrix for line segments')
    print('This will take a minute')
    start = time.time()
    
    n_edge = len(edges)
    M = np.zeros((n_edge,2,2),dtype=float)
    
    x_node = nodes[edges,0] #(n_edge,2)
    y_node = nodes[edges,1] #(n_edge,2)
    
    for i in range(n_edge):
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
        
        M[i,:,:] = R[:-1,:-1]
    
    #let
    # [x_old,y_old]^T be 2 x 1 vector
    # [x_new,y_new]^T be 2 x 1 vector
    # R[:-1,:-1] be 2 x 2 matrix
    #then
    # [x_new,y_new]^T = R[:-1,:-1].dot([x_old,y_old]^T)
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')

    return M

def grad_2d(f_n,elements,elem_mids,Je):
    print('Computing fields and gradients in elements')
    start = time.time()

    n_elem = len(elem_mids)
    f_out = np.zeros((n_elem,4),dtype=f_n.dtype)
    
    f_node = f_n[elements] #(n_elem,3)
    x_r = np.c_[elem_mids[:,0],elem_mids[:,0],elem_mids[:,0]] #(nelem,3)
    y_r = np.c_[elem_mids[:,1],elem_mids[:,1],elem_mids[:,1]] #(nelem,3)
    f_out[:,0] = np.sum((Je[:,:,0]+Je[:,:,1]*x_r+Je[:,:,2]*y_r)*f_node,axis=1)
    f_out[:,1] = np.sum(f_node*Je[:,:,1],axis=1)
    f_out[:,2] = np.sum(f_node*Je[:,:,2],axis=1)
    
    elapsed=time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    
    return f_out

def assemble_Ke2d(mesh,materials):
    print('Assembling the system of equations for triangular elements')
    print('This will take a while')
    start = time.time()
    n_node = len(mesh['nodes'])
    n_elem = len(mesh['elements'])
    n_rep = materials['c_x'].shape[1]
    
    nodes = mesh['nodes']
    elements = mesh['elements']
    Je = mesh['elem_shape_fun']
    area = mesh['elem_area']
    elem_proc = mesh['elem_proc']
    
    c_x = materials['c_x']
    c_y = materials['c_y']
    alpha_x = materials['alpha_x']
    alpha_y = materials['alpha_y']
    beta_x = materials['beta_x']
    beta_y = materials['beta_y']
    gamma_x = materials['gamma_x']
    gamma_y = materials['gamma_y']
    a = materials['a']
    f = materials['f']
    a_n = materials['a_n']
    f_n = materials['f_n']
    
    I = np.zeros(n_elem*9*n_rep**2,dtype=int)
    J = np.zeros(n_elem*9*n_rep**2,dtype=int)
    V1 = np.zeros(n_elem*9*n_rep**2,dtype=float)
    V2 = np.zeros(n_elem*9*n_rep**2,dtype=float)
    b1 = np.zeros(n_node*n_rep,dtype=float)
    b2 = np.zeros(n_node*n_rep,dtype=float)

    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(3*n_rep,dtype=int),3*n_rep,1).T
    COL = np.matlib.repmat(np.arange(3*n_rep,dtype=int),3*n_rep,1)
    
    pattern_k = np.any(np.sign(c_x).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(c_y).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(alpha_x).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(alpha_y).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(beta_x).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(beta_y).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(a).astype(bool),axis=0)    
    
    pattern_b = np.any(np.sign(f).astype(bool),axis=0)
    pattern_b = pattern_b|np.any(np.sign(gamma_x).astype(bool),axis=0)
    pattern_b = pattern_b|np.any(np.sign(gamma_y).astype(bool),axis=0)
    pattern_b = pattern_b|np.any(np.sign(f_n).astype(bool),axis=0)
    pattern_b = np.reshape(pattern_b,(-1,1))

    stack_k = [[None]*3 for i in range(3)]
    stack_b = [[None]*1 for i in range(3)]
    for i in range(3):
        for j in range(3):
            stack_k[i][j] = pattern_k
        stack_b[i][0] = pattern_b
    stack_k = np.asarray(np.bmat(stack_k))
    stack_b = np.asarray(np.bmat(stack_b))
    
    ind_k = np.where(stack_k.flatten(order='C'))[0]
    ind_b = np.where(stack_b.flatten(order='C'))[0]
    pattern_k = csr_matrix(pattern_k.astype(int))
    pattern_b = csr_matrix(pattern_b.astype(int))
    print('Sparsity pattern for Ke and be (zoom-out vs zoom-in)')
    fig,ax = plt.subplots(1,4,figsize=(8,2))
    axes = ax.flatten()
    axes[0].spy(stack_k)
    axes[1].spy(stack_b)
    axes[2].spy(pattern_k)
    axes[3].spy(pattern_b)
    axes[1].set_xticks(range(1))
    axes[3].set_xticks(range(1))
    axes[0].set_xlabel('Ke')
    axes[1].set_xlabel('be')
    axes[2].set_xlabel('Ke')
    axes[3].set_xlabel('be')
    plt.show()

    elem_proc = elem_proc&materials['inside']
    for i in range(n_elem):
        cnt = i*9*n_rep**2
        ind_n = elements[i,:]
        if elem_proc[i]==True:
            #Ke1,Ke2,be1 = build_Ke2d(c_x=c_x[i,:,:],c_y=c_y[i,:,:],
            #    alpha_x=alpha_x[i,:,:],alpha_y=alpha_y[i,:,:],
            #    beta_x=beta_x[i,:,:],beta_y=beta_y[i,:,:],
            #    gamma_x=gamma_x[i,:],gamma_y=gamma_y[i,:],
            #    a=a[i,:,:],f=f[i,:],a_n=a_n[ind_n,:,:],f_n=f_n[ind_n,:],
            #    Je=Je[i,:,:],area=area[i])
            Ke1,Ke2,be1 = quick_build_Ke2d(c_x=c_x[i,:,:],c_y=c_y[i,:,:],
                alpha_x=alpha_x[i,:,:],alpha_y=alpha_y[i,:,:],
                beta_x=beta_x[i,:,:],beta_y=beta_y[i,:,:],
                gamma_x=gamma_x[i,:],gamma_y=gamma_y[i,:],
                a=a[i,:,:],f=f[i,:],a_n=a_n[ind_n,:,:],f_n=f_n[ind_n,:],
                Je=Je[i,:,:],area=area[i],ind_k=ind_k,ind_b=ind_b)
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
                a_n,f_n,Je,area):
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
            #           a_n[jj,kk,ll]*(1+delta)/12.0)*area
            Ke1[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+c_y[kk,ll]*Je[ii,2]*Je[jj,2]+
                        (alpha_x[kk,ll]*Je[ii,1]+alpha_y[kk,ll]*Je[ii,2])/3.0+
                        (beta_x[kk,ll]*Je[jj,1]+beta_y[kk,ll]*Je[jj,2])/3.0)*area
            Ke2[i,j] = (a[kk,ll]+a_n[jj,kk,ll])*area*(1+delta)/12.0
        
        be1[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2]+f[kk]/3.0)*area

        for jj in range(3):
            delta = 1-np.abs(np.sign(ii-jj))
            be1[i] = be1[i]+f_n[jj,kk]*area*(1+delta)/12.0

    return Ke1,Ke2,be1

def quick_build_Ke2d(c_x,c_y,alpha_x,alpha_y,beta_x,beta_y,gamma_x,gamma_y,a,f,
               a_n,f_n,Je,area,ind_k,ind_b):
    n_rep = len(c_x)
    Ke1 = np.zeros((3*n_rep,3*n_rep),dtype=float)
    Ke2 = np.zeros((3*n_rep,3*n_rep),dtype=float)
    be1 = np.zeros(3*n_rep,dtype=float)
    
    for ij in ind_k:
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
        #           a_n[jj,kk,ll]*(1+delta)/12.0)*area
        Ke1[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+c_y[kk,ll]*Je[ii,2]*Je[jj,2]+
                    (alpha_x[kk,ll]*Je[ii,1]+alpha_y[kk,ll]*Je[ii,2])/3.0+
                    (beta_x[kk,ll]*Je[jj,1]+beta_y[kk,ll]*Je[jj,2])/3.0)*area
        Ke2[i,j] = (a[kk,ll]+a_n[jj,kk,ll])*area*(1+delta)/12.0
        
    
    for i in ind_b:
        ii = int(i/n_rep) #ii^th node, i = 1,2,3
        kk = int(i)%n_rep #kk^th unknown, j = 1,2,3,4,...,n_rep
        be1[i] = (gamma_x[kk]*Je[ii,1]+gamma_y[kk]*Je[ii,2]+f[kk]/3.0)*area
        for jj in range(3):
            delta = 1-np.abs(np.sign(ii-jj))
            be1[i] = be1[i]+f_n[jj,kk]*area*(1+delta)/12.0
    
    return Ke1,Ke2,be1

def assemble_Ke1d(mesh,materials):
    print('Assembling the system of equations for line segments')
    start = time.time()
    n_node = len(mesh['nodes'])
    n_edge = len(mesh['edges'])
    n_rep = materials['c_x'].shape[1]
    
    nodes = mesh['nodes']
    edges = mesh['edges']
    Je = mesh['edge_shape_fun']
    length = mesh['edge_len']
    edge_proc = mesh['edge_proc']
        
    c_x = materials['c_x']
    alpha_x = materials['alpha_x']
    beta_x = materials['beta_x']
    gamma_x = materials['gamma_x']
    a = materials['a']
    f = materials['f']

    I = np.zeros(n_edge*4*n_rep**2,dtype=int)
    J = np.zeros(n_edge*4*n_rep**2,dtype=int)
    V1 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    V2 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    b1 = np.zeros(n_node*n_rep,dtype=float)
    b2 = np.zeros(n_node*n_rep,dtype=float)
    
    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1).T
    COL = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1)

    pattern_k = np.any(np.sign(c_x).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(alpha_x).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(beta_x).astype(bool),axis=0)
    pattern_k = pattern_k|np.any(np.sign(a).astype(bool),axis=0)
    
    pattern_b = np.any(np.sign(f).astype(bool),axis=0)
    pattern_b = pattern_b|np.any(np.sign(gamma_x).astype(bool),axis=0)
    pattern_b = np.reshape(pattern_b,(-1,1))
    
    stack_k = [[None]*2 for i in range(2)]
    stack_b = [[None]*1 for i in range(2)]
    for i in range(2):
        for j in range(2):
            stack_k[i][j] = pattern_k
        stack_b[i][0] = pattern_b
    stack_k = np.asarray(np.bmat(stack_k))
    stack_b = np.asarray(np.bmat(stack_b))

    ind_k = np.where(stack_k.flatten(order='C'))[0]
    ind_b = np.where(stack_b.flatten(order='C'))[0]
    pattern_k = csr_matrix(pattern_k.astype(int))
    pattern_b = csr_matrix(pattern_b.astype(int))
    print('Sparsity pattern for Ke and be (zoom-out vs zoom-in)')
    fig,ax = plt.subplots(1,4,figsize=(8,2))
    axes = ax.flatten()
    axes[0].spy(stack_k)
    axes[1].spy(stack_b)
    axes[2].spy(pattern_k)
    axes[3].spy(pattern_b)
    axes[1].set_xticks(range(1))
    axes[3].set_xticks(range(1))
    axes[0].set_xlabel('Ke')
    axes[1].set_xlabel('be')
    axes[2].set_xlabel('Ke')
    axes[3].set_xlabel('be')
    plt.show()
    
    edge_proc = edge_proc&materials['inside']
    for i in range(n_edge):
        cnt = i*4*n_rep**2
        ind_n = edges[i,:]
        if edge_proc[i]==True:
            #Ke1,Ke2,be1 = build_Ke1d(c_x=c_x[i,:,:],alpha_x=alpha_x[i,:,:],beta_x=beta_x[i,:,:],
            #    gamma_x=gamma_x[i,:],a=a[i,:,:],f=f[i,:],Je=Je[i,:,:],length=length[i])
            Ke1,Ke2,be1 = quick_build_Ke1d(c_x=c_x[i,:,:],alpha_x=alpha_x[i,:,:],beta_x=beta_x[i,:,:],
                gamma_x=gamma_x[i,:],a=a[i,:,:],f=f[i,:],Je=Je[i,:,:],length=length[i],
                ind_k=ind_k,ind_b=ind_b)
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
            #          (beta_x[kk,ll]*Je[jj,1])/2.0)*length
            Ke1[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+
                        (alpha_x[kk,ll]*Je[ii,1])/2.0+
                        (beta_x[kk,ll]*Je[jj,1])/2.0)*length
            Ke2[i,j] = a[kk,ll]*length*(1+delta)/6.0
            
        be1[i] = (gamma_x[kk]*Je[ii,1]+f[kk]/2.0)*length

    return Ke1,Ke2,be1

def quick_build_Ke1d(c_x,alpha_x,beta_x,gamma_x,a,f,Je,length,ind_k,ind_b):
    n_rep = len(c_x)
    Ke1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    Ke2 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    be1 = np.zeros(2*n_rep,dtype=float)
    
    for ij in ind_k:
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
        #          (beta_x[kk,ll]*Je[jj,1])/2.0)*length
        Ke1[i,j] = (c_x[kk,ll]*Je[ii,1]*Je[jj,1]+
                    (alpha_x[kk,ll]*Je[ii,1])/2.0+
                    (beta_x[kk,ll]*Je[jj,1])/2.0)*length
        Ke2[i,j] = a[kk,ll]*length*(1+delta)/6.0
    
    for i in ind_b:
        ii = int(i/n_rep) #ii^th node, i=1,2,3
        kk = int(i)%n_rep #kk^th unknown, j=1,2,3,4,...,n_rep
        be1[i] = (gamma_x[kk]*Je[ii,1]+f[kk]/2.0)*length

    return Ke1,Ke2,be1

def assemble_Ks2d(mesh,materials):
    print('Incoorprating the boundary condition of the third kind')
    start = time.time()
    n_node = len(mesh['nodes'])
    n_edge = len(mesh['edges'])
    n_rep = materials['g_s'].shape[1]

    nodes = mesh['nodes']
    edges = mesh['edges']
    length = mesh['edge_len']
    edge_proc = mesh['edge_proc']

    g_s = materials['g_s']
    q_s = materials['q_s']
    

    I = np.zeros(n_edge*4*n_rep**2,dtype=int)
    J = np.zeros(n_edge*4*n_rep**2,dtype=int)
    V1 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    V2 = np.zeros(n_edge*4*n_rep**2,dtype=float)
    b1 = np.zeros(n_node*n_rep,dtype=float)
    b2 = np.zeros(n_node*n_rep,dtype=float)
    
    REP = np.reshape(np.arange(n_node*n_rep,dtype=int),(n_node,n_rep))
    ROW = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1).T
    COL = np.matlib.repmat(np.arange(2*n_rep,dtype=int),2*n_rep,1)
    
    pattern_k = np.any(np.sign(q_s).astype(bool),axis=0)    
    pattern_b = np.any(np.sign(g_s).astype(bool),axis=0)
    pattern_b = np.reshape(pattern_b,(-1,1))

    stack_k = [[None]*2 for i in range(2)]
    stack_b = [[None]*1 for i in range(2)]
    for i in range(2):
        for j in range(2):
            stack_k[i][j] = pattern_k
        stack_b[i][0] = pattern_b
    stack_k = np.asarray(np.bmat(stack_k))
    stack_b = np.asarray(np.bmat(stack_b))

    ind_k = np.where(stack_k.flatten(order='C'))[0]
    ind_b = np.where(stack_b.flatten(order='C'))[0]
    pattern_k = csr_matrix(pattern_k.astype(int))
    pattern_b = csr_matrix(pattern_b.astype(int))
    print('Sparsity pattern for Ks and bs (zoom-out vs zoom-in)')
    fig,ax = plt.subplots(1,4,figsize=(8,2))
    axes = ax.flatten()
    axes[0].spy(stack_k)
    axes[1].spy(stack_b)
    axes[2].spy(pattern_k)
    axes[3].spy(pattern_b)
    axes[1].set_xticks(range(1))
    axes[3].set_xticks(range(1))
    axes[0].set_xlabel('Ke')
    axes[1].set_xlabel('be')
    axes[2].set_xlabel('Ke')
    axes[3].set_xlabel('be')
    plt.show()

    edge_proc = edge_proc&materials['with_3rd_kind_bc']
    for i in range(n_edge):
        cnt = i*4*n_rep**2
        ind_n = edges[i,:]
        if edge_proc[i]==True:
            #Ks1,bs1 = build_Ks2d(g_s=g_s[i,:],q_s=q_s[i,:,:],length=length[i])
            Ks1,bs1 = quick_build_Ks2d(g_s=g_s[i,:],q_s=q_s[i,:,:],length=length[i],
                ind_k=ind_k,ind_b=ind_b)
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

def quick_build_Ks2d(g_s,q_s,length,ind_k,ind_b):
    n_rep = len(g_s)
    Ks1 = np.zeros((2*n_rep,2*n_rep),dtype=float)
    bs1 = np.zeros(2*n_rep,dtype=float)
    
    for ij in ind_k:
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

def set_1st_kind_bc(mesh,materials,K_in,b_in):
    print('Incoorprating the Dirichlet boundary condition')
    start = time.time()
    n_node = len(mesh['nodes'])
    n_rep = materials['s_n'].shape[1]
    
    nodes = mesh['nodes']
    edges = mesh['edges']
    
    on_1st_kind_bc = materials['on_1st_kind_bc'].flatten(order='C')
    s_n = materials['s_n'].flatten(order='C')
    
    K = csr_matrix.copy(K_in) #consider without copying
    b = np.array(b_in) #consider without copying
    
    b[~on_1st_kind_bc] = b[~on_1st_kind_bc]-K.dot(s_n)[~on_1st_kind_bc]
    b[on_1st_kind_bc] = s_n[on_1st_kind_bc]
    
    ind_n = np.where(on_1st_kind_bc)[0]
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

def solve_sparse(K,b):
    print('Calling sparse linear system solver')
    start = time.time()
    K.eliminate_zeros()
    sol = spsolve(K,b)
    elapsed = time.time()-start
    print('Time elapsed ',elapsed,'sec')
    print('')
    return sol

def update_materials(domain,stern,robin,dirichlet):
    print('Calling subroutine update_materials()')
    print('Please build your own subroutine to customize the material properties')
    print('')
    return

def run_simulation(mesh,physics,survey,dist_factor,run_mode):
    sigma_solid = physics['sigma_solid']
    ratio = survey['ratio']
    freq = survey['freq']
    sol_stat = [np.zeros(0)]*len(ratio)
    sol_pert = [[np.zeros(0)]*len(freq) for i in range(len(ratio))]
    if run_mode==0:
        domain = set_domain(mesh,physics,dist_factor,run_mode)
        stern = set_stern(mesh,physics,dist_factor,run_mode)
        robin = set_robin(mesh,physics,dist_factor,run_mode)
        dirichlet = set_dirichlet(mesh,physics,dist_factor,run_mode)

        #customize material properties if needed
        update_materials(domain,stern,robin,dirichlet)

        K1,K2,b1,b2 = assemble_Ke2d(mesh,domain)
        K3,K4,b3,b4 = assemble_Ke1d(mesh,stern)
        K5,K6,b5,b6 = assemble_Ks2d(mesh,robin)
    
        for i in range(len(ratio)):
            sigma_diffuse = -(1-ratio[i])*sigma_solid
            K = K1+K2+K3+K4+K5+K6
            b = b1+b2+b3+b4+b5*sigma_diffuse+b6
            K,b = set_1st_kind_bc(mesh,dirichlet,K,b)
    
            sol = solve_sparse(K,b)
            sol_stat[i] = sol
    
    return sol_stat,sol_pert

