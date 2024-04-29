import galsim
import batsim
import numpy as np
import argparse
from time import time

from tqdm import tqdm, trange

def main(args):
    '''
    Generate LSST-like images from parametric COSMOS
    galaxies with IA and PSF.
    '''
    n_gals = args.n_gals
    nn = args.nn
    scale = args.scale
    save_dir = args.save_dir

    # Create LSST-like PSF
    seeing = 0.67
    psf = galsim.Moffat(beta=3.5, fwhm=seeing, trunc=seeing*4)

    # Generate the galaxy objects
    cosmos_cat = galsim.COSMOSCatalog()

    # If no number provided, default to full COSMOS catalogue
    if n_gals is None:
        n_gals = len(cosmos_cat)

    gal_inds = np.random.choice(len(cosmos_cat), n_gals, replace=False)
    records = cosmos_cat.getParametricRecord(gal_inds)
    galaxies = cosmos_cat.makeGalaxy(index=gal_inds, gal_type='parametric')

    # Calculate the number of rotated galaxies
    n_rot = 4
    ng_eff = n_rot*n_gals
    rescale = np.random.uniform(0.5, 1.5, ng_eff)

    # Total size of stamp for 4 galaxies
    stamp_size = n_rot * nn

    progress = tqdm(total=ng_eff, desc='Generating images')
    for i in range(n_gals):
        
        gal = galaxies[i]
        # Expand the galaxy
        gal = gal.expand(rescale[i])

        ident = records[i]['IDENT']

        # Get HLR for galaxy to create IA transform
        if records[i]['use_bulgefit']:
            hlr = records[i]['hlr'][2]
        else:
            hlr = records[i]['hlr'][0]

        # Create IA transform
        IATransform = batsim.IaTransform(
            hlr=hlr,
            A=0.00136207,
            beta=0.82404653, # best first Georgiou19+
            phi = np.radians(0),
            clip_radius=3 # clip the transform at 5*hlr to prevent edge effects
        )

        rotated_gals = cancel_shape_noise(gal, n_rot)
        stamp = galsim.ImageF(stamp_size, stamp_size, scale=scale)
        for i in range(n_rot):

            # set drawing location on image
            row = i // int(np.sqrt(1*n_rot))
            col = i % int(np.sqrt(1*n_rot))

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

        # Save the image
        stamp.write(
            file_name='COSMOS_{}_noiseless.fits'.format(ident),
            dir=save_dir
        )

    progress.close()

    return

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
    parser = argparse.ArgumentParser(description='Generate noiseless LSST-like images from COSMOS galaxies')
    parser.add_argument('--n_gals', type=int, default=None, help='Number of galaxies to simulate')
    parser.add_argument('--nn', type=int, default=64, help='Number of pixels on a side')
    parser.add_argument('--scale', type=float, default=0.2, help='Pixel scale in arcseconds')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save images')

    args = parser.parse_args()

    main(args)






