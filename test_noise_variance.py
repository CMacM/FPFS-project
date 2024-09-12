import anacal
import galsim
import numpy as np
import matplotlib.pyplot as plt
import batsim
import astropy.io.fits as fits
import gc
import torch
import argparse

from tqdm import tqdm, trange

import jax
import jax.numpy as jnp
from jax import random, jit

from multiprocessing import Pool, cpu_count

def find_spaced_squares(start, end, N):
    # Take the square roots of start and end, rounding up and down
    sqrt_start = int(np.ceil(np.sqrt(start)))
    sqrt_end = int(np.floor(np.sqrt(end)))
    
    # Generate N linearly spaced values between the square roots (as integers)
    spaced_roots = np.linspace(sqrt_start, sqrt_end, N).astype(int)
    
    # Square those values to get perfect squares
    spaced_squares = np.square(spaced_roots)
    
    return spaced_squares

# now try inverse back into a list of single images
def split_image_into_quadrants(image, n_rot, nn):
    '''
    THIS CODE WILL SPLIT IMAGES OF 4 ROTATIONS INTO SINGLE
    IMAGES, EACH CONTAINING A SINGLE ROTATION. FOR SOME REASON
    THIS IS REQUIRED FOR THE BELOW METHOD OF MEASURING SHEAR.
    '''
    quadrants = []

    sqrt_n_rot = int(np.sqrt(n_rot))
    
    for j in range(n_rot):
        # Calculate the row and column positions
        row = j // sqrt_n_rot
        col = j % sqrt_n_rot

        # Calculate the bounds for this quadrant
        xmin = col * nn
        xmax = (col + 1) * nn
        ymin = row * nn
        ymax = (row + 1) * nn
        
        # Extract the quadrant
        quadrant = image[ymin:ymax, xmin:xmax]
        quadrants.append(quadrant)
    
    return quadrants

def main(args):

    # Ensure that PyTorch uses GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # force cpu
    #device = torch.device("cpu")

    print("Using device: {}".format(device))

    # Parameters
    n_min = args.n_min
    n_max = args.n_max
    n_samples = args.n_samples

    # find n_samples square numbers up to n_gals
    # Start at n_gals and work backwards
    n_gals_per_sample = find_spaced_squares(n_min, n_max, n_samples)

    filename = 'simulations/isolated/COSMOS_ngals=81499_noiseless.fits'
    cosmos = galsim.COSMOSCatalog()
    records = cosmos.getParametricRecord(np.arange(len(cosmos)))

    do_force_detect = True # Force to have a detection at the center of the image
    buff = 20

    noise_seed_base = args.seed
    noise_std = 0.37 # 0.37 is 10 year, * sqrt(10) for 1 year
    if args.year == 1:
        noise_std *= np.sqrt(10)
    noise_variance = noise_std ** 2.0

    i = 0
    ia_cosmos_list = []
    ia_cosmos_scenes = []
    hlrs = []
    use_bulge = []
    fluxes = []
    records_inds = []
    gal_ids = []

    n_iter = args.n_iter
    a_vals = np.empty((n_iter,n_samples))
    progress = tqdm(total=n_iter*n_samples, desc="Iterations", position=0)
    for a in range(n_iter):
        add_noise = True
        noise_seed = noise_seed_base + a
        # on the first iteration don't add noise
        if a == 0:
            add_noise = False
        with fits.open(filename) as hdul:
            for i in range(n_samples):
                scene_list = []
                for j in range(n_gals_per_sample[i]):
                    # Exclude the first HDU
                    data = hdul[j+1].data

                    # Split the image into quadrants
                    n_grid = int(data.shape[0] // 2)
                    data_quadrants = split_image_into_quadrants(data, 4, n_grid)
                    scene_list.extend(data_quadrants)  # Append to scene list

                    # Store a few other things about the galaxy
                    gal_id = hdul[j+1].header['IDENT']
                    gal_ids.append(gal_id)

                    record_ind = np.where(records['IDENT'] == gal_id)[0][0]
                    records_inds.append(record_ind)

                    bulge = records['use_bulgefit'][record_ind]
                    use_bulge.append(bulge)
                    if bulge:
                        hlrs.append(records['hlr'][record_ind][2])
                        fluxes.append(records['flux'][record_ind][3])
                    else:
                        hlrs.append(records['hlr'][record_ind][0])
                        fluxes.append(records['flux'][record_ind][0])

                # Append the scene list to the list of scenes
                ia_cosmos_list.append(scene_list)

                # Stack quadrants into a single image using PyTorch
                n = np.sqrt(len(scene_list))
                if n % 1 != 0:
                    print("Number of quadrants is not a square number. Padding with empty quadrants.")
                    n = np.ceil(n).astype(int)
                    # determine number of empty quadrants to add
                    # to make the number of quadrants a square number
                    n_empty = n**2 - len(scene_list)
                    # Append empty quadrants
                    for _ in range(n_empty):
                        scene_list.append(np.zeros((n_grid, n_grid)))   
                else:
                    n = int(n)

                with torch.no_grad():
                    scene = torch.zeros((n * n_grid, n * n_grid), device=device)
                    for k in range(n):
                        for l in range(n):
                            # Ensure native byte order for the NumPy array
                            quadrant = np.ascontiguousarray(scene_list[k*n + l].astype(np.float32))
                            scene[k*n_grid:(k+1)*n_grid, l*n_grid:(l+1)*n_grid] = torch.tensor(quadrant, device=device)

                    if do_force_detect:
                        pass
                    else:
                        # Pad the scene using PyTorch
                        scene = torch.nn.functional.pad(scene, (buff, buff, buff, buff), mode='constant', value=0)

                    if add_noise:
                        torch.manual_seed(noise_seed)
                        noise = torch.normal(mean=0.0, std=noise_std, size=scene.shape, device=device)
                        scene = scene + noise

                        torch.manual_seed(int(noise_seed + 1e6))
                        noise_array = torch.normal(mean=0.0, std=noise_std, size=scene.shape, device=device)
                        noise_array = noise_array.cpu().numpy()
                        del noise
                    else:
                        noise_array = None

                    ia_cosmos_scenes.append(scene.cpu().numpy())  # Move back to CPU and convert to NumPy array if necessary
                    del scene, scene_list
                
                # Flush the GPU memory
                torch.cuda.empty_cache()
                gc.collect()

                pixel_scale = 0.2
                ngrid = 64

                seeing = 0.8
                psf_obj = galsim.Moffat(beta=2.5, fwhm=seeing, trunc=seeing*4.0)
                psf_array = (
                    psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
                    .drawImage(nx=ngrid, ny=ngrid, scale=pixel_scale)
                    .array
                )

                fpfs_config_outer = anacal.fpfs.FpfsConfig(
                    sigma_arcsec=0.52, # detection kernel
                    sigma_arcsec2=1.0 # measurement kernel
                )

                fpfs_config_inner = anacal.fpfs.FpfsConfig(
                    sigma_arcsec=0.52, # detection kernel
                    sigma_arcsec2=0.45 # measurement kernel
                )

                nstamp = np.sqrt(n_gals_per_sample[i]).astype(int)
                if do_force_detect:
                    indx = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
                    indy = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
                    ns = len(indx) * len(indy)
                    inds = np.meshgrid(indy, indx, indexing="ij")
                    yx = np.vstack([np.ravel(_) for _ in inds])
                    buff = 0
                    dtype = np.dtype(
                        [
                            ("y", np.int32),
                            ("x", np.int32),
                            ("is_peak", np.int32),
                            ("mask_value", np.int32),
                        ]
                    )
                    coords = np.empty(ns, dtype=dtype)
                    coords["y"] = yx[0]
                    coords["x"] = yx[1]
                    coords["is_peak"] = np.ones(ns)
                    coords["mask_value"] = np.zeros(ns)
                else:
                    coords = None

                # Measurement
                output_outer = []
                output_inner = []
                output_outer.append(
                    anacal.fpfs.process_image(
                        mag_zero=30,
                        fpfs_config=fpfs_config_outer,
                        gal_array=ia_cosmos_scenes[0],
                        psf_array=psf_array,
                        pixel_scale=pixel_scale,
                        noise_variance=max(noise_variance, 0.23),
                        noise_array=noise_array,
                        coords=coords
                    )
                )

                output_inner.append(
                    anacal.fpfs.process_image(
                        mag_zero=30,
                        fpfs_config=fpfs_config_inner,
                        gal_array=ia_cosmos_scenes[0],
                        psf_array=psf_array,
                        pixel_scale=pixel_scale,
                        noise_variance=max(noise_variance, 0.23),
                        noise_array=noise_array,
                        coords=coords
                    )
                )

                del noise_array

                # Extract the measurements
                ename = "e1_2"
                egname = "e1_g1_2"
                wgname = "w_g1"

                e1_0 = output_outer[0]["w"] * output_outer[0][ename]
                e1g1_0 = (
                        output_outer[0][wgname] 
                        * output_outer[0][ename] 
                        + output_outer[0]["w"] 
                        * output_outer[0][egname]
                    )

                g1_outer = np.sum(e1_0) / np.sum(e1g1_0)

                e1_1 = output_inner[0]["w"] * output_inner[0][ename]
                e1g1_1 = (
                        output_inner[0][wgname] 
                        * output_inner[0][ename] 
                        + output_inner[0]["w"] 
                        * output_inner[0][egname]
                    )

                g1_inner = np.sum(e1_1) / np.sum(e1g1_1)

                a_vals[a,i] = g1_inner/g1_outer
                progress.update(1)

    progress.close()

    # Save results
    savename = args.savename
    if savename is None:
        savename = "a_vals_niter={}_seed={}_year={}.npz".format(n_iter, noise_seed_base, args.year)

    np.savez(
        savename, 
        a_vals=a_vals, 
        samples=n_gals_per_sample,
        inner=fpfs_config_inner.sigma_arcsec2, 
        outer=fpfs_config_outer.sigma_arcsec2
        )
    
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate isolated COSMOS galaxies with FPFS.")
    parser.add_argument("--n_min", type=int, default=100, help="Min number of galaxies to generate.")
    parser.add_argument("--n_max", type=int, default=81499, help="Max number of galaxies to generate.")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to test up to n_gals.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for noise generation.")
    parser.add_argument("--year", type=int, default=1, help="Number of years of noise.")
    parser.add_argument("--savename", type=str, default=None, help="Filename to save results to.")

    args = parser.parse_args()
    main(args)