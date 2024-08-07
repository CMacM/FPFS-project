import galsim
import batsim
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import fpfs
import pickle
from time import time

from tqdm import tqdm, trange

from argparse import ArgumentParser

# Set matplotlib parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10']
plt.rcParams['mathtext.fontset'] ='cm'
plt.rcParams['figure.facecolor'] = 'white'
plt.rc('axes', unicode_minus=False)
plt.rc('axes.formatter', use_mathtext=True)

def main(args):

    # Drawing parameters
    scale = args.scale
    nn = args.nn
    ngals = args.ngals

    # Galaxy catalogue and sample set-up
    cosmos_cat = galsim.COSMOSCatalog()
    gal_inds = np.random.choice(len(cosmos_cat), ngals)
    gal_sample = cosmos_cat.makeGalaxy( 
                                    gal_type='parametric', 
                                    noise_pad_size=0,
                                    index=gal_inds
                                    )

    # Lensing parameters
    gamma1 = 0.0006810353158702314
    gamma2 = 0.0
    kappa = 0.0

    g1 = gamma1 / (1 - kappa)
    g2 = gamma2 / (1 - kappa)
    mu = 1 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)

    # BATSim lensing transform object
    lens = batsim.LensTransform(gamma1=gamma1, gamma2=gamma2, kappa=kappa)

    # Create rotated sample
    nrot = 4

    # Total image size to contain all gals
    scene_nn = int(nn * np.sqrt(nrot))

    # PSF set-up
    seeing = 0.67
    psf_obj = galsim.Moffat(beta=3.5, fwhm=seeing, flux=1.0, trunc=4*seeing)

    rcut = 16
    psf_data = psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(nx=rcut*2, ny=rcut*2, scale=scale, method='auto')

    progress_bar = tqdm(total=ngals*nrot, desc='Drawing galaxies')
    for i, obj in enumerate(gal_sample):

        galsim_image = galsim.ImageF(scene_nn, scene_nn, scale=scale)
        batsim_image = galsim.ImageF(scene_nn, scene_nn, scale=scale)

        rotated_gals = cancel_shape_noise(obj, nrot)

        for j, gal in enumerate(rotated_gals):

            # set drawing location on image
            row = j // int(np.sqrt(nrot))
            col = j % int(np.sqrt(nrot))

            # Compute the bounds for this galaxy
            xmin = col * nn + 1  # +1 because GalSim coordinates start at 1
            xmax = (col + 1) * nn
            ymin = row * nn + 1
            ymax = (row + 1) * nn

            # shear using galsim
            lensed_gal = gal.lens(g1=g1, g2=g2, mu=mu)
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
                gal_obj=gal,
                transform_obj=lens,
                psf_obj=psf_obj,
                draw_method="auto"
            )

            bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
            sub_image = galsim.Image(bat_img, scale=scale)
            batsim_image.setSubImage(bounds, sub_image)

            progress_bar.update(1)

        if i == 0:
            batsim_scene = batsim_image.array
            galsim_scene = galsim_image.array
        else:
            batsim_scene = np.concatenate((batsim_scene, batsim_image.array), axis=1)
            galsim_scene = np.concatenate((galsim_scene, galsim_image.array), axis=1)

    progress_bar.close()

    # Test on a range of kernel sizes
    kernels = np.linspace(0.3,0.7,20)

    galsim_bias = np.empty(len(kernels))
    batsim_bias = np.empty(len(kernels))

    for i in trange(len(kernels)):
        # measure on galsim galaxies
        _, m_bias = test_kernel_size(
            sigma_arcsec=kernels[i], 
            psf_arr=psf_data.array, 
            gal_scene=galsim_scene,
            true_shear=g1,
            nx=nn,
            ny=nn,
            scale=scale,
        )
        galsim_bias[i] = m_bias

        # measure on batsim galaxies
        _, m_bias = test_kernel_size(
            sigma_arcsec=kernels[i], 
            psf_arr=psf_data.array, 
            gal_scene=batsim_scene, 
            true_shear=g1,
            nx=nn,
            ny=nn,
            scale=scale
        )
        batsim_bias[i] = m_bias

    # Plot results
    plt.plot(kernels, abs(batsim_bias), label='BATSim')
    plt.plot(kernels, abs(galsim_bias), label='Galsim')
    plt.legend()
    plt.xlabel('Shapelet scale (arcsec)')
    plt.ylabel('Multiplicative bias')
    plt.yscale('log')
    plt.title('Stamp size: %d pixels, Num. galaxies: %d'%(nn, ngals*nrot))
    plt.savefig('kernel_size_bias-stamp%d-ngals%d.png'%(nn,ngals*nrot), dpi=300)

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
    pickle.dump(report, open('kernel_size_bias-stamp%d-ngals%d.pkl'%(nn,ngals*nrot), 'wb'))

    return

# FPFS measurement function
def test_kernel_size(sigma_arcsec, psf_arr, gal_scene, true_shear, scale, nx, ny):

    # initialize FPFS shear measurement task
    fpTask = fpfs.image.measure_source(psf_arr, sigma_arcsec=sigma_arcsec, pix_scale=scale)

    scene_nx = gal_scene.shape[0]
    scene_ny = gal_scene.shape[1]
    
    # force detection in specific order 
    indX = np.arange(int(nx/2), scene_nx, nx)
    indY = np.arange(int(ny/2), scene_ny, ny)
    inds = np.meshgrid(indY, indX, indexing="ij")
    coords = np.vstack([np.ravel(_) for _ in inds]).T
    coords = np.flip(coords, axis=1)    

    # measure shear with FPFS on entire scene
    mms = fpTask.measure(gal_scene, coords)
    mms = fpTask.get_results(mms)

    # convert momemnts to ellipticity estimates
    ells = fpfs.catalog.fpfs_m2e(mms, const=2000)
    resp = np.average(ells['fpfs_R1E'])
    shear = np.average(ells['fpfs_e1'])/resp
    m_bias = abs(shear - true_shear)/true_shear

    return shear, m_bias

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scale', type=float, default=0.2)
    parser.add_argument('--nn', type=int, default=64)
    parser.add_argument('--ngals', type=int, default=10)
    args = parser.parse_args()
    main(args)
