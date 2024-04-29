import galsim
import batsim
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import fpfs
import pickle
from time import time

from tqdm import tqdm, trange

# Drawing parameters
scale = 0.2
nn = 256

# Galaxy catalogue and sample set-up
cosmos_cat = galsim.COSMOSCatalog()
ngals = 1000
gal_inds = np.random.choice(len(cosmos_cat), ngals)
gal_sample = cosmos_cat.makeGalaxy( 
                                   gal_type='parametric', 
                                   noise_pad_size=0,
                                   index=gal_inds
                                )

# Lensing parameters
gamma1 = 0.02
gamma2 = 0.0
kappa = 0.0

g1 = gamma1 / (1 - kappa)
g2 = gamma2 / (1 - kappa)
mu = 1 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)

# BATSim lensing transform object
lens = batsim.LensTransform(gamma1=gamma1, gamma2=gamma2, kappa=kappa)

def cancel_shape_noise(gal_obj, nrot):
    '''Create nrot rotated versions of the input galaxy object
    such that shape noise cancels out when averaging the shapes'''
    rotated_gals = []
    for i in range(nrot):
        rot_ang = np.pi / nrot * i
        ang = rot_ang * galsim.radians
        gal_obj = gal_obj.rotate(ang)
        rotated_gals.append(gal_obj)
        
    return rotated_gals

# Create rotated sample
nrot = 4
rotated_gals = []
for gal in tqdm(gal_sample):
    rotated_gals.extend(cancel_shape_noise(gal, nrot))

# Total image size to contain all gals
scene_nn = int(nn * np.sqrt(ngals*nrot))

# PSF set-up
seeing = 0.67
psf_obj = galsim.Moffat(beta=3.5, fwhm=seeing, flux=1.0, trunc=4*seeing)

rcut = 16
psf_data = psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(nx=rcut*2, ny=rcut*2, scale=scale, method='auto')

galsim_image = galsim.ImageF(scene_nn, scene_nn, scale=scale)
batsim_image = galsim.ImageF(scene_nn, scene_nn, scale=scale)
for i, obj in tqdm(enumerate(rotated_gals)):

    # set drawing location on image
    row = i // int(np.sqrt(ngals*nrot))
    col = i % int(np.sqrt(ngals*nrot))

    # Compute the bounds for this galaxy
    xmin = col * nn + 1  # +1 because GalSim coordinates start at 1
    xmax = (col + 1) * nn
    ymin = row * nn + 1
    ymax = (row + 1) * nn

    # shear using galsim
    lensed_gal = obj.lens(g1=g1, g2=g2, mu=mu)
    smeared_gal = galsim.Convolve([lensed_gal, psf_obj])
    bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
    sub_image = galsim_image[bounds]
    smeared_gal.shift(0.5*scale, 0.5*scale).drawImage(
        image=sub_image,
        add_to_image=True,
        method='auto'
        )

    #shear using batsim
    bat_img = batsim.simulate_galaxy(
        ngrid=nn,
        pix_scale=scale,
        gal_obj=obj,
        transform_obj=lens,
        psf_obj=psf_obj
    )
    bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
    sub_image = galsim.Image(bat_img, scale=scale)
    batsim_image.setSubImage(bounds, sub_image)

# FPFS measurement function
def test_kernel_size(sigma_arcsec, psf_arr, gal_arr, scale, rcut):

    # initialize FPFS shear measurement task
    fpTask = fpfs.image.measure_source(psf_arr, sigma_arcsec=sigma_arcsec, pix_scale=scale)

    p1 = scene_nn //2 - rcut
    p2 = scene_nn //2 - rcut
    psf_arr_pad = np.pad(psf_arr, ((p1, p1), (p2, p2)))

    coords = fpTask.detect_sources(gal_arr,psf_arr_pad,thres=0.01,thres2=-0.00)
    # measure shear with FPFS on individual galaxies
    mms = fpTask.measure(gal_arr, coords)
    mms = fpTask.get_results(mms)

    # convert momemnts to ellipticity estimates
    ells = fpfs.catalog.fpfs_m2e(mms, const=2000)
    resp = np.average(ells['fpfs_R1E'])
    shear = np.average(ells['fpfs_e1'])/resp
    shear_err = np.std(ells["fpfs_e1"]) / np.abs(resp) / np.sqrt(nrot*ngals)
    m_bias = abs(shear - g1)/g1

    return shear, shear_err, m_bias

# Test on a range of kernel sizes
# Test different kernels
# Test different kernels
kernels = np.linspace(0.2,1.4,20)

galsim_bias = np.empty(len(kernels))
batsim_bias = np.empty(len(kernels))

for i in trange(len(kernels)):
    # measure on galsim galaxies
    _, _, m_bias = test_kernel_size(
        sigma_arcsec=kernels[i], 
        psf_arr=psf_data.array, 
        gal_arr=galsim_image.array, 
        scale=scale, 
        rcut=rcut
    )
    galsim_bias[i] = m_bias

    # measure on batsim galaxies
    _, _, m_bias = test_kernel_size(
        sigma_arcsec=kernels[i], 
        psf_arr=psf_data.array, 
        gal_arr=batsim_image.array, 
        scale=scale,
        rcut=rcut
    )
    batsim_bias[i] = m_bias

# Plot results
plt.plot(kernels, abs(batsim_bias), label='BATSim')
plt.plot(kernels, abs(galsim_bias), label='Galsim')
plt.hline()
plt.legend()
plt.xlabel('Kernel Size')
plt.ylabel('Multiplicative Bias')
plt.yscale('log')
plt.title('Stamp size: %d pixels'%nn)
plt.savefig('kernel_size_bias-stamp%d-ngals%d.png'%(nn,1000*nrot), dpi=300)

# Collate parameters for test report
best_kernel = kernels[np.argmin(batsim_bias)]
worst_kernel = kernels[np.argmax(batsim_bias)]
exceeds_lsst = kernels[np.where(batsim_bias > 0.013)[0]]

# Save test report
report = {
    'scale': scale,
    'nn': nn,
    'ngals': ngals*nrot,
    'kernels': kernels,
    'best_kernel': best_kernel,
    'worst_kernel': worst_kernel,
    'exceeds_lsst': exceeds_lsst,
    'batsim_bias': batsim_bias,
    'galsim_bias': galsim_bias
}
pickle.dump(report, open('kernel_size_bias-stamp%d-ngals%d.pkl'%(nn,1000*nrot), 'wb'))
