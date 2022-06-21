import numpy as np
from lib.mesh import Mesh,Probe
from lib.materials import Consts,PertPNP,Physics
from lib.assembly import PertFEM
from lib.utils import print_tstamp

mesh_prefix = 'capsol/probe'
dist_factor = 1e6
height_water = 18e-9
offset_tip = 20e-9

probe = Probe(
    #geometry of the background slab
    radius_air = 1000e-6, #radius of the air
    height_air = 1000e-6, #height of the air
    radius_water = 1000e-6, #radius of thin water film
    height_water = height_water, #thickness of thin water film
    radius_solid = 1000e-6, #radius of the solid
    height_solid = 1000e-6, #height of the solid

    #geometry of the cantilever probe
    radius_tip = 20e-9, #radius of probe tip
    offset_tip = offset_tip, #offset between probe tip and sw interface
    radius_cone = 15e-6*np.tan(15/180*np.pi)+20e-9, #radius of probe cone
    height_cone = 15e-6, #height of probe cone
    radius_disk = 35e-6, #radius of probe disk
    height_disk = 0.5e-6, #height of probe disk

    mesh_prefix = mesh_prefix,
    dist_factor = dist_factor,
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
    s_0 = 10.0, #voltage on equipotential surface
    radius_a = 0.0, #radius of sphere particle
    is_solid_metal = False, #True if solid is metal otherwise false
    )

height_water = [5e-9,10e-9,20e-9]
rel_perm_a = [80.0,80.0,1.0]
offset_tip = [100e-9]

#for i in range(len(height_water)):
for i in [0]:
    for j in range(len(offset_tip)):
        print('*'*80)
        h = height_water[i]*1e9
        s = offset_tip[j]*1e9
        mesh_prefix = 'capsol/probe_h{0:.1f}nm_s{1:.1f}nm'.format(h,s)

        #update probe
        probe.height_water = height_water[i]
        probe.offset_tip = offset_tip[j]
        probe.mesh_prefix = mesh_prefix
        probe.build_mesh = True
        probe.build()

        #import mesh
        mesh = Mesh(
            prefix = mesh_prefix,
            axis_symmetry = 'Y',
            unscale_factor=1/dist_factor,
            )

        #build PDE using physics
        physics.rel_perm_a = rel_perm_a[i]
        pnp = PertPNP(physics)

        #build FEM system
        pert = PertFEM(mesh,pnp)
        time,freq,ft,ftarg = pert.argft(freqtime=np.logspace(-6,2,301),
                                        signal=1,ft='dlf',ftarg={})
        #pert.ftsolve(ratio=[1.0],freqtime=np.logspace(-6,1,301),
        #             signal=1,ft='dlf',ftarg={},n_proc=1) #wrapped
        #pert.ftsolve(ratio=[1.0],freqtime=freq,stat=None,n_proc=1)
        pert.solve(ratio=[1.0],freq=freq,stat=None,n_proc=1)
        pert.save(mesh_prefix)
        print_tstamp()

        # #display results
        # mesh.tripcolor(np.real(slab.fsol[0][0,:,-2]),cmap='YlGnBu_r')
        # mesh.tripcolor(np.imag(slab.fsol[0][0,:,-2]),cmap='YlGnBu_r')

