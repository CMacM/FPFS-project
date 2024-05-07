import galsim
import batsim
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import fpfs
from time import time

from tqdm import tqdm, trange

# Obtain a variety of galaxy profiles from COSMOS
ngal = 250
seed = 42

galaxy_catalog = galsim.COSMOSCatalog()
np.random.seed(seed)
rands = np.random.randint(0, len(galaxy_catalog), ngal)
gal_sample = galaxy_catalog.makeGalaxy(
    index=rands,
    gal_type='parametric',
    noise_pad_size=0,
    gsparams=galsim.GSParams(maximum_fft_size=10000)
    )
records = galaxy_catalog.getParametricRecord(index=rands)

hlrs = []
fluxes = []
for i in range(ngal):
    if records['use_bulgefit'][i]:
        hlrs.append(records['hlr'][i][2])
        fluxes.append(records['flux'][i][3])
    else:
        hlrs.append(records['hlr'][i][0])
        fluxes.append(records['flux'][i][0])

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

nrot = 4
nn = 64
scale = 0.2

a_ia = 0.00136207
b_ia = 0.82404653

# Create LSST-like PSF
seeing = 0.67
psf = galsim.Moffat(beta=3.5, fwhm=seeing, trunc=seeing*4)

stamp_size = int(nn * np.sqrt(nrot))

progress = tqdm(total=ngal*nrot, desc='Generating images')

image_list = []
for i in range(ngal):
    
    gal = gal_sample[i]

    # Create IA transform
    IATransform = batsim.IaTransform(
        scale=scale,
        hlr=hlrs[i],
        A=a_ia,
        beta=b_ia, # best first Georgiou19+
        phi = np.radians(0),
        clip_radius=3 # clip the transform at 5*hlr to prevent edge effects
    )

    if i == 0:
        g1, g2 = IATransform.get_g1g2(hlrs[i],0)
        print(f"True g1: {g1}, True g2: {g2}")

    rotated_gals = cancel_shape_noise(gal, nrot)
    stamp = galsim.ImageF(stamp_size, stamp_size, scale=scale)
    for i in range(nrot):

        # set drawing location on image
        row = i // int(np.sqrt(nrot))
        col = i % int(np.sqrt(nrot))

        # Compute the bounds for this galaxy
        xmin = col * nn + 1  # +1 because GalSim coordinates start at 1
        xmax = (col + 1) * nn
        ymin = row * nn + 1
        ymax = (row + 1) * nn

        #Convolve with PSF and pixel response
        gal_img = batsim.simulate_galaxy(
            ngrid=nn,
            pix_scale=scale,
            gal_obj=rotated_gals[i],
            transform_obj=IATransform,
            psf_obj=psf,
            draw_method="auto"
        )

        # Set the subimage in the stamp
        bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
        sub_image = galsim.Image(gal_img, scale=scale)
        stamp.setSubImage(bounds, sub_image)
        
        # Update progress bar
        progress.update(1)

    image_list.append(stamp.array)

progress.close()

# Force detection to save time

def test_kernel(nx, ny, psf_data, gal_data, scale, shapelet_kernel=0.1):

    # Force detection at the stamp center point (ngrid//2, ngrid//2)
    indX = np.arange(int(nx/2), nx*2, nx)
    indY = np.arange(int(ny/2), ny*2, ny)
    inds = np.meshgrid(indY, indX, indexing="ij")
    coords = np.vstack([np.ravel(_) for _ in inds]).T

    fpTask = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=shapelet_kernel
        )
    
    mms = fpTask.measure(gal_data, coords)
    mms = fpTask.get_results(mms)
    ells=  fpfs.catalog.fpfs_m2e(mms,const=2000)
    resp=np.average(ells['fpfs_R1E'])
    shear=np.average(ells['fpfs_e1'])/resp

    return shear, resp

psf_data = psf.shift(0.5*scale, 0.5*scale).drawImage(nx=nn, ny=nn, scale=scale)

kernels = np.linspace(0.3, 1.0, 20)
rwfs = np.zeros((len(kernels), ngal))

total = len(kernels) * ngal
progress = tqdm(total=total, desc='Computing RWFs')
for i in range(len(kernels)):
    for j in range(ngal):
        shear, resp = test_kernel(
            nn,
            nn,
            psf_data.array,
            image_list[j],
            scale,
            shapelet_kernel=kernels[i]
            )

        a_rwf = galsim.Shear(g1=shear).e1
        rwfs[i,j] = (a_rwf / a_ia) ** (1/b_ia)

        progress.update(1)
        
progress.close()

# Plot results
plt.figure()
for i in range(ngal):
    plt.plot(kernels, rwfs[:,i])
plt.xlabel('Shapelet kernel size (arcsec)')
plt.ylabel('Radial weight function (HLR)')
plt.savefig('plots/rwf_vs_kernel.png')

# Pick middle kernel size
i = 10
plt.figure()
plt.scatter(hlrs, rwfs[i,:], marker='o', c=rwfs[i,:], cmap='viridis')
plt.xlabel('Half light radius (arcsec)')
plt.ylabel('Radial weight function (HLR)')
plt.savefig('plots/rwf_vs_hlr.png')

plt.figure()
plt.scatter(fluxes, rwfs[i,:], lw=0, marker='o', c=rwfs[i,:], cmap='viridis')
plt.xlabel('Flux')
plt.ylabel('Radial weight function (HLR)')
plt.savefig('plots/rwf_vs_flux.png')

zphot = records['zphot']
plt.figure()
plt.scatter(zphot, rwfs[i,:], lw=0, marker='o', c=rwfs[i,:], cmap='viridis')
plt.xlabel('zphot')
plt.ylabel('Radial weight function (HLR)')
plt.savefig('plots/rwf_vs_zphot.png')

np.savez('rwfs_test_ngal%d_seed%d.npz'%(ngal,seed),
        rwfs=rwfs,
        records=records,
        hlrs=hlrs,
        fluxes=fluxes,
        kernels=kernels,
        kernel_i=i
    )