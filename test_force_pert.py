import numpy as np
import subprocess
from lib.mesh import Mesh,Probe,generate_grid
from lib.materials import Consts,PertPNP,Physics
from lib.assembly import PertFEM
from lib.utils import print_tstamp

def call_triangle(mesh_prefix,triangle_path):
    command = [triangle_path+' -pnAae '+mesh_prefix+'.poly']
    process = subprocess.Popen(command,shell=True)
    process.wait()

    return

#CapSol FD Grid
kwargs = {'n':500,'m':500,'l_js':350, #number of grids: x,z+,z- below sample top
          'h0':1.0,'rho_max':10e6,'z_max':10e6, #resolution: h0, box_size (rho_max,z_max)
          'd_min':20,'d_max':110,'id_step':1, #tip-sample separation: min,max,istep (stepsize=istep*h0)
          'eps_r':5.9,'Hsam':10e6} #sample: eps_r, thickness_sample
x,y = generate_grid(**kwargs)

#CapSol Probe Geometry
probe = Probe(
    #geometry of the background slab
    radius_air = 10e6, #radius of the air [nm]
    height_air = 10e6, #height of the air [nm]
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
    mesh_prefix = 'none',
    mesh_grid = [x,y],
    dist_factor = 1.0,
    build_mesh = False,
    )

#define physical parameters
physics = Physics(
    c_ion = [0.0,0.0], #ion concentration at infinity [mol/m^3]
    z_ion = [-1.0,1.0], #ion valence or charge number [SI]
    mu_a = [5e-8,5e-8], #ion mobility in electrolyte [m^2/(Vs)]
    mu_s = 5e-15, #ion mobility in stern [m^2/(Vs)]
    rel_perm_a = 80.0, #relative permittivity of electrolyte [SI]
    rel_perm_i = 4.5, #relative permittivity of solid [SI]
    sigma_solid = -0.01, #surface charge density for solid [C] #sigma_intrinsic
    temperature = 293.0, #ambient temperature [K]
    e_0 = [0.0,0.0], #Ex/Ey at infinity
    f_0 = [[0.0,0.0,0.0]], #x/y/charge density at point sources
    s_0 = 1.0, #voltage on equipotential surface
    radius_a = 0.0, #radius of sphere particle
    is_solid_metal = False, #True if solid is metal otherwise false
    )

height_gap = [20.0]
h0 = [1.0]
for i in range(len(height_gap)):
    mesh_prefix = 'capsol/pert_s{:02.1f}nm'.format(height_gap[i])
    print(mesh_prefix)

    #update grid discretization
    kwargs['d_min'] = height_gap[i]
    kwargs['h0'] = h0[i]
    x,y = generate_grid(**kwargs)

    #update probe
    probe.height_gap = height_gap[i]
    probe.mesh_prefix = mesh_prefix
    probe.mesh_grid = [x,y]
    probe.build_mesh = False
    probe.build()
    call_triangle(mesh_prefix,'triangle')

    #import mesh
    mesh = Mesh(
        prefix = mesh_prefix,
        axis_symmetry = 'Y',
        unscale_factor = 1e-9,
        )

    #build PDE using physics
    pde = PertPNP(physics)

    #build FEM system
    pert = PertFEM(mesh,pde)
    time,freq,ft,ftarg = pert.argft(freqtime=np.logspace(-6,2,301),
                                    signal=1,ft='dlf',ftarg={})
    #pert.ftsolve(ratio=[1.0],freqtime=np.logspace(-6,1,301),
    #             signal=1,ft='dlf',ftarg={},n_proc=1) #wrapped
    #pert.ftsolve(ratio=[1.0],freqtime=freq,stat=None,n_proc=1)
    pert.solve(ratio=[1.0],freq=freq[66:],stat=None,n_proc=1)
    #pert.save(mesh_prefix)
    print_tstamp()
