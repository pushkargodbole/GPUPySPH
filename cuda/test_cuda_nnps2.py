"""A simple example demonstrating the use of PySPH as a library for
working with particles.

The fundamental summation density operation is used as the
example. Specifically, a uniform distribution of particles is
generated and the expected density via summation is compared with the
numerical result.

This tutorial illustrates the following:

   - Creating particles : ParticleArray
   - Setting up a periodic domain : DomainManager
   - Nearest Neighbor Particle Searching : NNPS

"""

# PySPH imports
from pyzoltan.core.carray import UIntArray
from pysph.base import utils
from pysph.base.nnps import DomainManager, BoxSortNNPS, Cell, arange_uint
from pysph.base.kernels import CubicSpline, Gaussian, QuinticSpline, WendlandQuintic
from pysph.tools.uniform_distribution import uniform_distribution_cubic2D, \
    uniform_distribution_hcp2D, get_number_density_hcp

# NumPy
import numpy
import numpy as np

from math import ceil, floor

# Python timer
from time import time

# Pycuda imports
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
#from pycuda.tools import PageLockedMemoryPool
from pycuda.tools import DeviceMemoryPool

# particle spacings
dx = 0.1; dxb2 = 0.5 * dx
h0 = 2.*dx
max = 1.
min = 0.

is2D = True

if is2D:
    maxz = 0.
    minz = 0.
else:
    maxz = max
    minz = min

# Uniform lattice distribution of particles
x, y, dx, dy, xmin, xmax, ymin, ymax = uniform_distribution_cubic2D(
    dx, xmin=min, xmax=max, ymin=min, ymax=max)

# Uniform hexagonal close packing arrangement of particles
#x, y, dx, dy, xmin, xmax, ymin, ymax = uniform_distribution_hcp2D(
#    dx, xmin=0.0, xmax=1., ymin=0.0, ymax=1., adjust=True)

# SPH kernel
#k = CubicSpline(dim=2)
#k = Gaussian(dim=2)
#k = QuinticSpline(dim=2)
k = WendlandQuintic(dim=2)

# for the hexagonal particle spacing, dx*dy is only an approximate
# expression for the particle volume. As far as the summation density
# test is concerned, the value will be uniform but not equal to 1. To
# reproduce a density profile of 1, we need to estimate the kernel sum
# or number density of the distribution based on the kernel
wij_sum_estimate = get_number_density_hcp(dx, dy, k, h0)
volume = 1./wij_sum_estimate
print 'Volume estimates :: dx^2 = %g, Number density = %g'%(dx*dy, volume)

x = x.ravel(); y = y.ravel()
h = numpy.ones_like(x) * h0
m = numpy.ones_like(x) * volume
wij = numpy.zeros_like(x)

# use the helper function get_particle_array to create a ParticleArray
pa = utils.get_particle_array(x=x,y=y,h=h,m=m,wij=wij)

# the simulation domain used to request periodicity
domain = DomainManager(
    xmin=min, xmax=max, ymin=min, ymax=max,periodic_in_x=True, periodic_in_y=True)



# NNPS object for nearest neighbor queries
nps = BoxSortNNPS(dim=2, particles=[pa,], radius_scale=k.radius_scale, domain=domain)
cell_size = nps.cell_size
print cell_size

DMP = DeviceMemoryPool()
max_cell_pop_gpu, nc, num_particles = nps.get_max_cell_pop(0, DMP)

max_cell_pop = max_cell_pop_gpu.get()


cells_gpu = gpuarray.zeros((nc[0]*nc[1]*nc[2], int(max_cell_pop)), dtype=np.int32)-1
cellpop_gpu = gpuarray.zeros((nc[0], nc[1], nc[2]), dtype=np.int32)
#cells_gpu_ptr = np.intp(cells_gpu.base.get_device_pointer())

indices = arange_uint(num_particles)

cells = nps.bin_cuda(0, indices, cells_gpu, cellpop_gpu, max_cell_pop, num_particles, DMP)

print cells.shape

print cells[0]
print cells[1]

"""
t0 = time()

print pa.num_real_particles*27*max_cell_pop
nbrs_gpu = cuda.pagelocked_empty(shape=(pa.num_real_particles, 27*max_cell_pop), dtype=np.int32)
nbrs_gpu_ptr = np.intp(nbrs_gpu.base.get_device_pointer())

nnbrs_gpu = cuda.pagelocked_empty(shape=(pa.num_real_particles), dtype=np.int32)
nnbrs_gpu_ptr = np.intp(nnbrs_gpu.base.get_device_pointer())
print "Alloc time:", time()-t0

# container for neighbors
print "Cuda Start"
t1 = time()
nps.get_nearest_particles_cuda(0, 0, pa.num_real_particles, nbrs_gpu_ptr, nnbrs_gpu_ptr, max_cell_pop, DMP)
t2 = time()-t1
print "Cuda End"
print "NumPa:", pa.num_real_particles
nbrs = UIntArray()
print "Basic Start"
t1 = time()
for i in range(pa.num_real_particles):
    nps.get_nearest_particles(0, 0, i, nbrs)
    if i == pa.num_real_particles-1:
        print len(nbrs), nbrs.get_npy_array()
        print nnbrs_gpu[pa.num_real_particles-1], nbrs_gpu[pa.num_real_particles-1][:nnbrs_gpu[pa.num_real_particles-1]]
    
t3 = time()-t1
print "Basic End"    
print "****************************************"
print "Cuda Time =", t2
print "Basic Time =", t3
print nnbrs_gpu
nbrs_gpu.base.free()
nnbrs_gpu.base.free()
"""
