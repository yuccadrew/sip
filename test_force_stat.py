import numpy as np
from scipy.interpolate import interp1d

from lib.mesh import Mesh,Probe,generate_grid
from lib.materials import Consts,PDE
from lib.assembly import FEM
from lib.utils import print_tstamp

#===============================================================================
#CapSol Grid
inparg = {'n':500,'m':500,'l_js':350, #number of grids: x,z+,z- below sample top
          'h0':1,'rho_max':10e6,'z_max':10e6, #resolution: h0, box_size (rho_max,z_max)
          'd_min':2,'d_max':20,'id_step':1, #tip-sample separation: min,max,istep (stepsize=istep*h0)
          'eps_r':5.9,'Hsam':10e6} #sample: eps_r, thickness_sample
x,y = generate_grid(**inparg)

#Probe Geometry
probe = Probe(
    #geometry of the background slab
    radius_air = 10e6, #radius of the air
    height_air = 10e6, #height of the air
    height_gap = 20.0, #gap between tip apex and sample surface [nm]
    height_water = 1.0, #thickness of thin water film
    height_solid = 10e6, #height of the solid

    #geometry of the cantilever probe
    radius_tip = 20, #radius of probe tip [nm]
    radius_cone = 15e3*np.tan(15/180*np.pi)+20, #radius of probe cone [nm]
    height_cone = 15e3+20, #height of probe cone [nm]
    radius_disk = 35e3, #radius of probe disk [nm]
    height_disk = 0.5e3, #height of probe disk [nm]

    #area constraints
    area_air = 1e12, #[nm]**2
    area_water = 1e12, #[nm]**2
    area_solid = 1e12, #[nm]**2

    #mesh construction parameters
    mesh_prefix = 'test',
    mesh_grid = [x,y],
    dist_factor = 1.0,
    build_mesh = False,
    )

#define PDE
pde = PDE(
    c_x = {'is_in_air':[[1.0]],
           'is_in_water':[[1.0]],
           'is_in_solid':[[5.9]]},
    c_y = {'is_in_air':[[1.0]],
           'is_in_water':[[1.0]],
           'is_in_solid':[[5.9]]},
    s_n = {'is_on_inner_bound':[1.0],
           'is_on_top_bound':[0.0],
           'is_on_bottom_bound':[0.0],
           'is_on_right_bound':[0.0],
           'is_on_outside_domain':[0.0]},
    )

#===============================================================================
height_gap = np.arange(2,20+1,1)

# for i in range(len(height_gap)):
for i in range(0,len(height_gap)):
    mesh_prefix = 'capsol/test4_s{:02.1f}nm'.format(height_gap[i])
    #print(mesh_prefix)

    #update grid discretization
    inparg['d_min'] = height_gap[i]
    x,y = generate_grid(**inparg)

    #update probe geometry and control points
    probe.height_gap = height_gap[i]
    probe.mesh_prefix = mesh_prefix
    probe.mesh_grid = [x,y]
    probe.build_mesh = True
    probe.build()

    #update PDE boundary condition
    f_name = 'fort.{0:.0f}00'.format(1e3+height_gap[i])
    d_in = np.genfromtxt('capsol/grid_500_4/'+f_name)
    r = np.reshape(d_in[:,0],(len(x),-1))[:,0]
    z = np.reshape(d_in[:,1],(len(x),-1))[0,:]

    #boundary condition at top
    z_ind = np.argmin(abs(z-max(probe.cpts[:,1])))
    pot_top = d_in[z_ind::len(z),2]
    print('Top Bound:',np.unique(d_in[z_ind::len(z),1]),max(probe.cpts[:,1]))

    #boundary condition at bottom
    z_ind = np.argmin(abs(z-min(probe.cpts[:,1])))
    pot_bot = d_in[z_ind::len(z),2]
    print('Bottom Bound:',np.unique(d_in[z_ind::len(z),1]),min(probe.cpts[:,1]))

    #boundary condition at right side
    r_ind = np.argmin(abs(r-max(probe.cpts[:,0])))
    pot_rs = d_in[r_ind*len(z):r_ind*len(z)+len(z),2]
    print('Right Bound:',np.unique(d_in[r_ind*len(z):r_ind*len(z)+len(z),0]),
          max(probe.cpts[:,0]))

#     def build_s_top(x,y,*args):
#         return interp1d(r,pot_top)(x)

#     def build_s_bot(x,y,*args):
#         return interp1d(r,pot_bot)(x)

#     def build_s_rs(x,y,*args):
#         return interp1d(z,pot_rs)(y)

#     pde.s_n['is_on_top_bound'] = [build_s_top]
#     pde.s_n['is_on_bottom_bound'] = [build_s_bot]
#     pde.s_n['is_on_right_bound'] = [build_s_rs]

    #import mesh
    mesh = Mesh(
        prefix = mesh_prefix,
        axis_symmetry = 'Y',
        unscale_factor = 1.0,
        )

    #build FEM system
    stat = FEM(mesh,pde)
    stat.solve()
    stat.save(mesh_prefix)
    print_tstamp()
