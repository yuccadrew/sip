import numpy as np
import datetime
import subprocess
from .mesh import Flags,import_nodes,import_elements,import_edges

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

def build_probe(probe,mesh_prefix,disp_flag=False):
#     #user inputs of slab geometry
#     radius_air = 1000e-6 #radius of the air
#     height_air = 1000e-6 #height of the air
#     radius_water = radius_air #radius of thin water film
#     height_water = 10e-9 #thickness of thin water film
#     radius_solid = radius_air #radius of the solid
#     height_solid = height_air #height of the solid

#     #user inputs of probe geometry
#     radius_tip = 20e-9 #radius of probe tip
#     offset_tip = 100e-9 #offset between probe tip and sw interface
#     radius_cone = 15e-6*np.tan(15/180*np.pi)+radius_tip #radius of probe cone
#     height_cone = 15e-6 #height of probe cone
#     radius_disk = 35e-6 #radius of probe disk
#     height_disk = 0.5e-6 #height of probe disk

    #user inputs of background geometry
    radius_air = probe.radius_air #radius of the air
    height_air = probe.height_air #height of the air
    radius_water = probe.radius_air #radius of thin water film
    height_water = probe.height_water #thickness of thin water film
    radius_solid = probe.radius_air #radius of the solid
    height_solid = probe.height_air #height of the solid

    #user inputs of probe geometry
    radius_tip = probe.radius_tip #radius of probe tip
    offset_tip = probe.offset_tip #offset between probe tip and sw interface
    radius_cone = probe.radius_cone #radius of probe cone
    height_cone = probe.height_cone #height of probe cone
    radius_disk = probe.radius_disk #radius of probe disk
    height_disk = probe.height_disk #height of probe disk

    #discretize rho
    lambda_d = min(9e-9,height_water)
    rho_min = 0
    rho_max = radius_solid
    rho = discretize_rho(lambda_d,rho_min,rho_max)

    #insert air-water interface into the discretization
    mask = rho<height_water
    rho = np.r_[rho[mask],height_water,rho[~mask]]
    #ind = np.argmin((rho-height_water)**2)
    #rho[ind] = height_water

    #print out discretization
    print('See radial discretization below')
    print(rho)

    #hard-coded mesh indexing constants
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

    #generate mesh
    #This script only works for the same height and radius for air and solid
    #X is the axis of symmetry
    #Y is the longitudinal axis
    cpts = np.zeros((0,3)) #coord_x/coord_y/flag of control points
    segs = np.zeros((0,3)) #ind_a/ind_b/flag of line segmenets
    holes = np.zeros((0,2)) #coord_x/coord_y
    zones = np.zeros((0,3)) #coord_x/coord_y/area

    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
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

    #--------------------------------------------------------------------------
    #define the top edge points of the solid (top right corner point included)
    #skip edge points on the axis of symmetry
    x = np.r_[rho[1:-1],radius_solid]
    y = np.r_[0*rho[1:-1],0]
    cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

    #--------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------
    #define the edge points on the tip surface
    #skip edge points on the axis of symmetry
    nA = 32
    ns = nA+1-2
    dA = np.pi/nA
    phi = np.arange(1,ns+1)*dA-np.pi/2 #half the circle
    x = radius_tip*np.cos(phi)+0.0
    y = radius_tip*np.sin(phi)+offset_tip+radius_tip
    cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

    #--------------------------------------------------------------------------
    #define the edge points on the cone surface
    x = np.r_[radius_cone]
    y = np.r_[height_cone]+offset_tip+2*radius_tip
    cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0

    #--------------------------------------------------------------------------
    #define inner control points along the cantilever
    x = np.r_[radius_disk,radius_disk]
    y = np.r_[0,height_disk]+offset_tip+2*radius_tip+height_cone
    cpts = np.r_[cpts,np.c_[x,y,np.ones(len(x))*0]] #node flag of 0
    
    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
    #define the segments on the lowermost boundary
    x = np.r_[0,radius_b]
    y = np.r_[-radius_b,-radius_b]
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.bottom_bound]] #predefined edge flag
    #print('Lowermost boundary: edge flag 1')

    #define the segments on the rightmost boundary
    x = np.r_[radius_b,radius_b]
    y = np.r_[-radius_b,radius_b]
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.right_bound]] #predefined edge flag
    #print('Rightmost boundary: edge flag 2')

    #define the segments on the topmost boundary
    x = np.r_[0,radius_b]
    y = np.r_[radius_b,radius_b]
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.top_bound]] #predefined edge flag
    #print('Uppermost boundary: edge flag 3')

    #define the segments on the leftmost boundary (axis of symmetry)
    mask = (rho>0)&(rho<offset_tip)
    y = np.r_[-radius_b,-np.flipud(rho),rho[mask],offset_tip,
              offset_tip+2*radius_tip,offset_tip+2*radius_tip+height_cone,
              offset_tip+2*radius_tip+height_cone+height_disk,radius_b]
    x = np.zeros_like(y)
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.axis_symmetry]] #predefined edge flag
    #print('Axis of symmetry: edge flag 4')

    #--------------------------------------------------------------------------
    #define the segments on the top edge of the solid (solid-liquid interface)
    x = np.r_[rho[:-1],radius_solid]
    y = np.zeros_like(x)
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.sw_interface]] #predefined edge flag
    #print('Solid-liquid interface: edge flag 7')

    #--------------------------------------------------------------------------
    #define the segments on the right edge of the water
    if radius_water<radius_air:
        x = np.r_[radius_water,radius_water]
        y = np.r_[0,height_water]
        for i in range(len(x)-1):
            ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
            ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
            segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.aw_interface]] #predefined edge flag
        #print('Right of water: edge flag 8')

    #define the segments on the top edge of the water
    #x = np.r_[0,radius_water]
    #y = np.r_[height_water,height_water]
    x = np.r_[rho[rho<radius_water],radius_water]
    y = np.zeros_like(x)+height_water
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.aw_interface]] #predefined edge flag
    #print('Air-water interface: edge flag 9')

    #--------------------------------------------------------------------------
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
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]] #predefined edge flag
    #print('Lower tip surface: edge flag 10')

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
#         segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]] #predefined edge flag
#     #print('Upper tip surface: edge flag 11')
    
    #--------------------------------------------------------------------------
    #define the right segments along the cone surface
    x = np.r_[radius_tip,radius_cone]
    y = np.r_[0,height_cone+radius_tip]+offset_tip+radius_tip
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]] #predefined edge flag
    #print('Right cone surface: edge flag 100')

#     #define the top segments along the cone surface
#     x = np.r_[0,radius_cone]
#     y = np.r_[height_cone,height_cone]+offset_tip+2*radius_tip
#     for i in range(len(x)-1):
#         ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
#         ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
#         segs = np.r_[segs,np.c_[ind_a,ind_b,100]] #edge flag of 100
#     #print('Top cone surface: edge flag 100')

    #--------------------------------------------------------------------------
    #define the segments along the remaining cantilever surface
    x = np.r_[0,radius_disk,radius_disk,radius_cone]
    y = np.r_[height_disk,height_disk,0,0]+offset_tip+2*radius_tip+height_cone
    for i in range(len(x)-1):
        ind_a = np.argmin((cpts[:,0]-x[i])**2+(cpts[:,1]-y[i])**2)
        ind_b = np.argmin((cpts[:,0]-x[i+1])**2+(cpts[:,1]-y[i+1])**2)
        segs = np.r_[segs,np.c_[ind_a,ind_b,Flags.equipotential_surf]] #predefined edge flag
    #print('Arm surface: edge flag 100')
    print('')

    #**************************************************************************
    #--------------------------------------------------------------------------
    #==========================================================================
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
    area = np.r_[100,1,100]
    zones = np.r_[zones,np.c_[x,y,area]]

    #build the poly file
    f1 = open(mesh_prefix+'.poly','w')
    f1.write(str(len(cpts))+'  2 0 1  #verticies #dimensions  #attributes #boundary markers \n')
    
    #write the vertices
    cnt = 1
    for i in range(len(cpts)):
        f1.write("{0:6.0F} {1:20.5F} {2:20.5F} {3:6.0F}\n".format(cnt,cpts[i,0]*1e6,cpts[i,1]*1e6,cpts[i,2]))
        cnt = cnt+1
    f1.write('\n')

    #write the segments
    cnt = 1
    f1.write(str(len(segs))+' 1 #segments, boundary marker\n')
    for i in range(len(segs)):
        f1.write("{0:6.0F} {1:5.0F} {2:5.0F} {3:6.0F}\n".format(cnt,segs[i,0]+1,segs[i,1]+1,segs[i,2]))
        cnt = cnt+1
    f1.write('\n')

    #write holes
    f1.write('%d\n'%(len(holes)))
    for i in range(len(holes)):
        x = holes[i,0]
        y = holes[i,1]
        f1.write('{0:6.0F} {1:12.6F} {2:12.6F} 1\n'.format(i+1,x*1e6,y*1e6))
    f1.write('\n')

    #write area constraints for zones
    f1.write('%d\n'%(len(zones)))
    for i in range(len(zones)):
        x = zones[i,0]
        y = zones[i,1]
        area = zones[i,2]
        f1.write('{0:6.0F} {1:12.6F} {2:12.6F} {3:6.0F} {4:12.6F}\n'.format(i+1,x*1e6,y*1e6,i+1,area))

    f1.write('\n')
    f1.write('# triangle -pnq30Aae '+mesh_prefix+'.poly \n')
    f1.close()

    #**************************************************************************
    #==========================================================================
    #--------------------------------------------------------------------------
    process = subprocess.Popen(['triangle -pnq30Aae '+mesh_prefix+'.poly'],shell=True)
    process.wait()

    #**************************************************************************
    #==========================================================================
    #--------------------------------------------------------------------------
    if disp_flag:
        disp_factor = 1e6
        fig,ax = plt.subplots(2,2,figsize=(8,8))
        axs = ax.flatten()
        x = cpts[segs[:,:-1].astype(int),0]
        y = cpts[segs[:,:-1].astype(int),1]
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

    #**************************************************************************
    #==========================================================================
    #--------------------------------------------------------------------------
    if disp_flag:
        nodes,node_flags = import_nodes(mesh_prefix)
        elements,elem_flags = import_elements(mesh_prefix)
        edges,edge_flags = import_edges(mesh_prefix)
        nodes = nodes*1e-6 #unscale nodes

        #display mesh
        disp_factor=1e6
        x=nodes[:,0]*disp_factor
        y=nodes[:,1]*disp_factor

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

    return

def discretize_rho(lambda_d,rho_min,rho_max):
    #obtain fine grid next to the solid-liquid interface
    #use 16 points between 0.02*debye_len and 10*debye_len
    #reduce lambda_d if water film is too thin
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
    #no need to refine diffuse layer because there is no diffuse layer
    #rho = np.r_[rho_min,rho_max]
    
    if False:
        #display and check the discretization
        fig,ax = plt.subplots()
        mask = rho>0
        ax.plot(rho[mask]/lambda_d,np.zeros_like(rho[mask]),'o')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\rho$/$\lambda_D$')
        #ax.set_xlim(0.01,20)
        plt.show()

    return rho

def print_tstamp():
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print('')
    return
