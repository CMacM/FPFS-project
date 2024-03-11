import numpy as np
import galsim
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

cosmos = galsim.COSMOSCatalog()

def process_galaxy(index, bigfft):
    try:
        galaxy = cosmos.makeGalaxy(index=index, gal_type='parametric', noise_pad_size=0, rng=None, gsparams=bigfft)
        radius = galaxy.calculateMomentRadius()
    except:
        radius = -1
    return index, radius

cosmax = len(cosmos)
bigfft = galsim.GSParams(maximum_fft_size=40000)

r_sec = np.empty(cosmax)

with ProcessPoolExecutor(5) as executor:
    # Submit all tasks
    futures = [executor.submit(process_galaxy, index, bigfft) for index in range(cosmax)]

    # Collect results as they complete
    results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=cosmax)]

# Sort results by index
results.sort(key=lambda x: x[0])

# Store sorted results in r_sec
for index, radius in results:
    r_sec[index] = radius

# r_sec now contains your sorted results
np.savez('cosmos_r_sec.npz', r_sec)


