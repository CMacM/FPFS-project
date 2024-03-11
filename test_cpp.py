import galsim
import batsim.transforms as batforms
import batsim.stamp as batstamp
import _gsinterface
import matplotlib.pyplot as plt

from time import time


# load in COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog()

# create a galaxy
rng = galsim.BaseDeviate(1000)
gal = cosmos_cat.makeGalaxy(n_random=1, rng=rng, 
                            gal_type='parametric')

# get simulation pixel scale and number
sim_scale = gal.nyquist_scale
sim_nn = gal.getGoodImageSize(sim_scale)
print("Sim scale: ", sim_scale)
print("Sim nn: ", sim_nn)

# time to create BATSim stamp
start = time()
stamp = batstamp.Stamp(nn=sim_nn, scale=sim_scale)
print(time()-start)
im = stamp.sample_galaxy(gal) 
print(time()-start)

gs_im = gal.drawImage(nx=sim_nn, ny=sim_nn, scale=sim_scale).array
plt.plot(im-gs_im)
plt.savefig("test.png")