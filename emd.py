import numpy as np
from scipy.spatial.distance import cdist
from ot.lp import emd_c, check_result

def process_event(event):
        # Sort constituents in event by pT
        event = np.array(event)
        event = event[event[:, 0].argsort()]
        pts = event[:, 0]
        coords = event[:, 1:]
        return pts, coords

def check_shape(event):
    assert (len(event.shape)) == 2 and (event.shape[-1] == 3) # Event shape must be (n, 3)

def emt_pot(source_event, target_event, R=1.0, return_flow=False, n_iter_max=100000):
    # Compute energy  mover's distance between two events using python ot library 
    
    check_shape(source_event)
    check_shape(target_event)

    source_pts, source_coords = process_event(source_event)
    target_pts, target_coords = process_event(target_event)

    # Psuedorapidity-azimuth distance matrix
    dist_matrix = cdist(source_coords, target_coords, 'euclidean')

    _, cost, flow, _, result_code = emd_c(source_pts, target_pts, dist_matrix, numItermax=n_iter_max, log=True)
    check_result(result_code)

    if return_flow:
        return cost, flow
    else:
        return cost


    

