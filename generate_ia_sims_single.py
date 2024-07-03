import galsim
import batsim
import numpy as np
import argparse
import pickle
import os

from multiprocessing import Pool, cpu_count
from parallelbar import progress_starmap
from tqdm import tqdm
from time import time

def generate_images(i, nn, scale, save_dir, galaxies, records, psf, n_rot, stamp_size):
    gal = galaxies[i]
    ident = records['IDENT'][i]

    if os.path.exists(os.path.join(save_dir, f'COSMOS_{ident}_noiseless.fits')):
        return
    
    # Get HLR for galaxy to create IA transform
    if records['use_bulgefit'][i]:
        hlr = records['hlr'][i][2]
    else:
        hlr = records['hlr'][i][0]

    # Create IA transform
    IATransform = batsim.IaTransform(
        scale=scale,
        hlr=hlr,
        A=0.00136207,
        beta=0.82404653, # best first Georgiou19+
        phi = np.radians(0),
        clip_radius=5 # clip the transform at 5*hlr to prevent edge effects
    )

    rotated_gals = cancel_shape_noise(gal, n_rot)
    stamp = galsim.ImageF(stamp_size, stamp_size, scale=scale)
    for j in range(n_rot):
        # set drawing location on image
        row = j // int(np.sqrt(1*n_rot))
        col = j % int(np.sqrt(1*n_rot))

        # Compute the bounds for this galaxy
        xmin = col * nn + 1  # +1 because GalSim coordinates start at 1
        xmax = (col + 1) * nn
        ymin = row * nn + 1
        ymax = (row + 1) * nn

        # Convolve with PSF and pixel response
        gal_img = batsim.simulate_galaxy(
            ngrid=nn,
            pix_scale=scale,
            gal_obj=rotated_gals[j],
            transform_obj=IATransform,
            psf_obj=psf,
            draw_method="auto"
        )

        # Set the subimage in the stamp
        bounds = galsim.BoundsI(xmin, xmax, ymin, ymax)
        sub_image = galsim.Image(gal_img, scale=scale)
        stamp[bounds] = sub_image

    # Save the image
    stamp.write(os.path.join(save_dir, f'COSMOS_{ident}_noiseless.fits'))

def main(args):
    '''
    Generate LSST-like images from parametric COSMOS
    galaxies with IA and PSF.
    '''

    start = time()

    n_gals = args.n_gals
    nn = args.nn
    scale = args.scale
    save_dir = args.save_dir
    batch_process = args.batch_process
    seed = args.seed
    n_proc = args.n_proc

    # Create LSST-like PSF
    seeing = 0.8
    psf = galsim.Moffat(beta=2.5, fwhm=seeing, trunc=seeing*4)

    # Generate the galaxy objects
    cosmos_cat = galsim.COSMOSCatalog()

    # If no number provided, default to full COSMOS catalogue
    if n_gals is None:
        n_gals = len(cosmos_cat)

    # Randomly select n_gals galaxies
    np.random.seed(seed)
    gal_inds = np.random.choice(len(cosmos_cat), n_gals, replace=False)
    records = cosmos_cat.getParametricRecord(gal_inds)
    galaxies = cosmos_cat.makeGalaxy(index=gal_inds, gal_type='parametric')

    # Calculate the number of rotated galaxies
    n_rot = 4
    ng_eff = n_rot*n_gals

    print(f'Generating {n_gals} images with {n_rot} rotations per image, {nn}x{nn} pixels, scale {scale} arcsec/pixel')

    # Total size of stamp for 4 galaxies
    stamp_size = int(nn * np.sqrt(n_rot))

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    if batch_process == 'True':
        print('Batch processing images')
        original_omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
        os.environ['OMP_NUM_THREADS'] = '1'

        if n_proc == None:
            n_proc = cpu_count()//2

        args_list = [(i, nn, scale, save_dir, galaxies, records, psf, n_rot, stamp_size) for i in range(n_gals)]
        progress_starmap(generate_images, args_list, n_cpu=n_proc)
    
        if original_omp_num_threads is None:
            del os.environ['OMP_NUM_THREADS']
        else:
            os.environ['OMP_NUM_THREADS'] = original_omp_num_threads
    elif batch_process == 'False':
        print('Looping through images')
        progress = tqdm(total=ng_eff, desc='Generating images')
        for i in range(n_gals):
            generate_images(i, nn, scale, save_dir, galaxies, records, psf, n_rot, stamp_size)
            progress.update(n_rot)
        progress.close()
    else:
        raise ValueError('batch_process must be either True or False')

    pickle.dump(psf, open(os.path.join(save_dir, 'psf.pkl'), 'wb'))

    print(f'Simulated {n_gals*n_rot} galaxies in {time() - start:.2f} seconds')


def cancel_shape_noise(gal_obj, nrot):
    '''Create nrot rotated versions of the input galaxy object
    such that shape noise cancels out when averaging the shapes'''
    rotated_gals = []
    for i in range(nrot):
        rot_ang = np.pi / nrot * i
        ang = rot_ang * galsim.radians
        rotated_gals.append(gal_obj.rotate(ang))
        
    return rotated_gals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate noiseless LSST-like images from COSMOS galaxies')
    parser.add_argument('--n_gals', type=int, default=None, help='Number of galaxies to simulate')
    parser.add_argument('--nn', type=int, default=64, help='Number of pixels on a side')
    parser.add_argument('--scale', type=float, default=0.2, help='Pixel scale in arcseconds')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save images')
    parser.add_argument('--batch_process', type=str, default=True, help='Batch process the images')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--n_proc', type=int, default=None, help='Number of processes to use')

    args = parser.parse_args()

    main(args)

