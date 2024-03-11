import galsim
import batsim.stamp as batstamp
import batsim.transforms as batforms
import batsim._gsinterface as _gsi

import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import fpfs
from time import time

from tqdm import tqdm, trange

cosmos_CAT = galsim.COSMOSCatalog()

ngals = len(cosmos_CAT)
gsparams = galsim.GSParams(maximum_fft_size=512*20, folding_threshold=1.e-3)
print("Number of galaxies in COSMOS catalog: ", ngals)

# Generate all galaxies
cosmos_gals = cosmos_CAT.makeGalaxy(index=np.arange(ngals), noise_pad_size=0, gal_type='parametric')

stamp_sz = np.empty((ngals))
for i, gal in tqdm(enumerate(cosmos_gals)):
    nq_scale = gal.nyquist_scale
    stamp_sz[i] = gal.getGoodImageSize(nq_scale)

large_stamps = np.where(stamp_sz > 10000)[0]
print("Number of large stamps: ", len(large_stamps))

largest = np.argmax(stamp_sz)
print("Largest stamp: ", largest, stamp_sz[largest])

# Select only smaller stamps
cosmos_gals_sel = np.delete(cosmos_gals, large_stamps)
stamp_sel = np.delete(stamp_sz, large_stamps)
print("Number of smaller stamps: ", len(cosmos_gals_sel))

# plot distribution of remaining stamp sizes
size_freq, size_bins = np.histogram(stamp_sel, bins=100)

# Estimate the runtime for the sample
rng = np.random.RandomState(1)
bin_runtime = np.empty((len(size_bins)))

diameter = 8.4  # meters
effective_diameter = 6.423  # meters, area weighted
seeing = 0.67  # arcseconds

AtmosphericPSF = galsim.Kolmogorov(fwhm=seeing, flux=1.0)
OpticalPSF = galsim.OpticalPSF(lam=500, diam=effective_diameter, obscuration=effective_diameter/diameter)

TotPSF = galsim.Convolve([AtmosphericPSF, OpticalPSF])
#TotPSF = AtmosphericPSF

gamma1 = 0.02
gamma2 = 0.
kappa = 0.
Lens = batforms.LensTransform(gamma1=gamma1, gamma2=gamma2, kappa=kappa, center=None)

times = []
chosen_sz = []
for i in trange(len(size_bins)-1):
    bin_inds = np.where((stamp_sel >= size_bins[i]) & (stamp_sel < size_bins[i+1]))[0]
    chosen_ind = rng.choice(bin_inds, 1)[0]
    gal = cosmos_gals_sel[chosen_ind]
    sim_scale = gal.nyquist_scale
    sim_nn = gal.getGoodImageSize(sim_scale)
    chosen_sz.append(sim_scale)

    # Estimate runtime
    start = time()

    # rotate and cancel shape noise
    nrot = 4
    rotated_gals = []
    for i in range(nrot):
        rot_ang = np.pi / nrot * i
        ang = rot_ang * galsim.radians
        gal = gal.rotate(ang)
        rotated_gals.append(gal)

    stamp = batstamp.Stamp(nn=sim_nn, scale=sim_scale, centering='fpfs')
    stamp.transform_grids(Lens)

    for obj in rotated_gals:
        batsim_gal = stamp.sample_galaxy(obj)
        batsim_im = galsim.Image(batsim_gal, scale=sim_scale)
        batsim_interp = galsim.InterpolatedImage(batsim_im, scale=sim_scale, normalization='sb')
        batsim_conv = galsim.Convolve([batsim_interp, TotPSF])
        batsim_final = batsim_conv.drawImage(nx=128, ny=128, scale=0.168, method='auto')
    
    end = time()
    times.append(end-start)
    
total_time = np.sum(times * size_freq)
print("Total runtime: ", total_time/(60**2 * 24), "days")

plt.plot(chosen_sz, times, 'o')
plt.xlabel('Stamp size / pixels')
plt.ylabel('Runtime / seconds')
plt.savefig('runtime_vs_stamp_size.png')