import numpy as np
from numpy.random import seed
seed(1)
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#%% Define functions

def import_data(nx,ny,n):
    folder = 'data_'+ str(nx) + '_' + str(ny)
    filename = './Results/'+folder+'/data_' + str(int(n))+'.npz'
    data = np.load(filename)
    w = data['w']
    s = data['s']
    t = data['t']
    return w,s,t

#%% Main program
# Inputs
lx = 8
ly = 1
#nx = 4096
nx = 1024
ny = int(nx/8)

Tm = 8
dt = 5e-4
nt = int(np.round(Tm/dt))
ns = 800
freq = int(nt/ns)

#%% grid

dx = lx/nx
dy = ly/ny

x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% Load POD data

folder = 'data_'+ str(nx) + '_' + str(ny)
filename = './POD/'+folder+'/POD_data.npz'
data = np.load(filename)

wm = data['wm']
Phiw = data['Phiw']
sm = data['sm']
Phis = data['Phis']
tm = data['tm']
Phit = data['Phit']
aTrue = data['aTrue']
bTrue = data['bTrue']

#%% Select the first nr
nr = 8 # define step; use subsampling a=a[start:stop:step]
nq = 16
Phiw = Phiw[:,:nq]
Phis = Phis[:,:nq]
Phit = Phit[:,:nq]
aTP = aTrue[:,:nq]
bTP = bTrue[:,:nq]

if not os.path.exists('./Plots/'):
    os.makedirs('./Plots/')

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

time = np.linspace(0,Tm,nt+1)
freq = 20
x_ticks = [0,2,4,6,8]

fig, axs = plt.subplots(4,2,figsize=(12,10))
axs= axs.flat
for k in range(8):
    axs[k].plot(time[::freq],aTP[:,k],label=r'\bf{FOM}', color = 'k', linewidth=3)
    axs[k].set_xticks(x_ticks)
    axs[k].set_xlabel(r'$t$')
    axs[k].set_ylabel(r'$\beta_{'+str(k+1)+'}(t)$')

axs[2].set_yticks([-100,0,100])
axs[7].set_yticks([-50,0,50])

axs[4].yaxis.labelpad = 12
axs[6].yaxis.labelpad = 12

axs[0].legend(loc="center", bbox_to_anchor=(1.18,1.3),ncol =4)#,fontsize=15)

fig.subplots_adjust(hspace=0.7, wspace=0.35)

plt.savefig('./Plots/BSbeta.png', dpi = 500, bbox_inches = 'tight')


#%% Load FOM results for t=0,2,4,8
n=0 #t=0
w0,s0,t0 = import_data(nx,ny,n)
n=int(2*nt/8) #t=2
w2,s2,t2 = import_data(nx,ny,n)
n=int(4*nt/8) #t=4
w4,s4,t4 = import_data(nx,ny,n)
n=int(8*nt/8) #t=8
w8,s8,t8 = import_data(nx,ny,n)

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
mpl.rc('font', **font)

nlvls = 31
x_ticks = [0,1,2,3,4,5,6,7,8]
y_ticks = [0,1]

#colormap = 'viridis'
#colormap = 'gnuplot'
#colormap = 'inferno'
colormap = 'seismic'

v = np.linspace(1.05, 1.45, nlvls, endpoint=True)
ctick = np.linspace(1.05, 1.45, 5, endpoint=True)

fig, axs = plt.subplots(4,1,figsize=(10,8.5))
axs= axs.flat

cs = axs[0].contour(X,Y,t0,v,cmap=colormap,linewidths=3)
cs.set_clim([1.05, 1.45])
cs = axs[1].contour(X,Y,t2,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[2].contour(X,Y,t4,v,cmap=colormap,linewidths=0.5)
cs.set_clim([1.05, 1.45])
cs = axs[3].contour(X,Y,t8,v,cmap=colormap,linewidths=0.5)#, rasterized=True)
cs.set_clim([1.05, 1.45])

for i in range(4):
    axs[i].set_xticks(x_ticks)
    axs[i].set_xlabel('$x$')
    axs[i].set_yticks(y_ticks)
    axs[i].set_ylabel('$y$')

# Add titles
fig.text(0.92, 0.83, '$t=0$', va='center')
fig.text(0.92, 0.63, '$t=2$', va='center')
fig.text(0.92, 0.43, '$t=4$', va='center')
fig.text(0.92, 0.23, '$t=8$', va='center')

fig.subplots_adjust(bottom=0.18, hspace=1)
cbar_ax = fig.add_axes([0.125, 0.03, 0.775, 0.045])
CB = fig.colorbar(cs, cax = cbar_ax, ticks=ctick, orientation='horizontal')
CB.ax.get_children()[0].set_linewidths(3.0)

plt.savefig('./Plots/BSFOM.png', dpi = 500, bbox_inches = 'tight')

# requirements on system:
# sudo apt-get install -y cm-super

#??? I got this error what is this font?
# findfont: Font family ['normal'] not found. Falling back to DejaVu Sans.


