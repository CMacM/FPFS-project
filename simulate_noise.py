# Script to add pixel noise to shape-noise cancelled COSMOS sims
import numpy as np
import galsim
import astropy.io.fits as astrofits
import fitsio
import argparse
from tqdm import tqdm

def main(args):
    '''
    Main function to add pixel noise to shape-noise cancelled COSMOS sims
    https://www.diva-portal.org/smash/get/diva2:813843/FULLTEXT01.pdf
    Some key numbers for LSST are:
    Gain = 0.34 e-/ADU
    Read noise = 10 e-
    Sky level = 98 (ADU/pixel) estimated
    '''
    # Read in galaxy images with IA
    filename = args.filename
    save_dir = args.save_dir
    noise_type = args.noise_type
    seed = args.seed

    rng = galsim.BaseDeviate(seed)

    if noise_type == 'gaussian':
        sigma = args.sigma
        noise = galsim.GaussianNoise(rng=rng, sigma=sigma)
    elif noise_type == 'poisson':
        sky_level = args.sky_level
        noise = galsim.PoissonNoise(rng=rng, sky_level=sky_level)
    elif noise_type == 'ccd':
        sky_level = args.sky_level
        gain = args.gain
        read_noise = args.read_noise
        noise = galsim.CCDNoise(rng=rng, gain=gain, read_noise=read_noise)

    n_gals = 81499
    n_scenes = 5
    gals_per_scene = n_gals // n_scenes

    total_indices = np.arange(0,n_gals)
    split_inds = np.array_split(total_indices, n_scenes)

    cosmos = galsim.COSMOSCatalog()
    records = cosmos.getParametricRecord(np.arange(len(cosmos)))

    i = 0
    ia_cosmos_list = []
    hlrs = []
    use_bulge = []
    fluxes = []
    records_inds = []
    gal_ids = []
    with astrofits.open(filename) as hdul:
        for j in tqdm(split_inds[i]):
            # exclude the first HDU
            data = hdul[j+1].data
            ia_cosmos_list.append(data)

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

        ia_cosmos_scene = np.concatenate(ia_cosmos_list, axis=1)