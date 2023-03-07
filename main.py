import sys
from glob import glob

import time
import h5py
import pygrib
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime, timedelta

sys.path.insert(0, '/glade/u/home/ksha/OPT_NCAR/')
sys.path.insert(0, '/glade/u/home/ksha/OPT_NCAR/libs/')

from namelist import *
import data_utils as du
import model_utils as mu

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('year', help='year')
parser.add_argument('mon', help='mon')
parser.add_argument('day', help='day')

args = vars(parser.parse_args())

# ===================================== #
date_temp = datetime(int(args['year']), int(args['mon']), int(args['day']), 0, 0)
# ===================================== #

N_lead = len(leads)
N_var = len(var_names)
half_margin = int(input_size/2)

with h5py.File(save_dir+'HRRRv4_STATS.hdf', 'r') as h5io:
    mean_stats = h5io['mean_stats'][...]
    std_stats = h5io['std_stats'][...]
    max_stats = h5io['max_stats'][...]
    
with h5py.File(save_dir+'HRRR_domain.hdf', 'r') as h5io:
    lon_3km = h5io['lon_3km'][...]
    lat_3km = h5io['lat_3km'][...]
    lon_80km = h5io['lon_80km'][...]
    lat_80km = h5io['lat_80km'][...]
    land_mask_80km = h5io['land_mask_80km'][...]

lon_80km_mask = lon_80km[land_mask_80km]
lat_80km_mask = lat_80km[land_mask_80km]

lon_minmax = [np.min(lon_80km_mask), np.max(lon_80km_mask)]
lat_minmax = [np.min(lat_80km_mask), np.max(lat_80km_mask)]

shape_80km = lon_80km.shape
shape_3km = lon_3km.shape

indx_array = np.empty(shape_80km)
indy_array = np.empty(shape_80km)

gridTree = cKDTree(list(zip(lon_3km.ravel(), lat_3km.ravel()))) #KDTree_wraper(xgrid, ygrid)

for xi in range(shape_80km[0]):
    for yi in range(shape_80km[1]):
        
        temp_lon = lon_80km[xi, yi]
        temp_lat = lat_80km[xi, yi]
        
        dist, indexes = gridTree.query(list(zip(np.array(temp_lon)[None], np.array(temp_lat)[None])))
        indx_3km, indy_3km = np.unravel_index(indexes, shape_3km)
        
        indx_array[xi, yi] = indx_3km[0]
        indy_array[xi, yi] = indy_3km[0]
        
# Crerate model
model = mu.create_model(input_shape=(input_size, input_size, N_var))

# get current weights
W_new = model.get_weights()

# get stored weights
print('Loading weights from {}'.format(model_name))
W_old = mu.dummy_loader(model_name)

# update stored weights to new weights
for i in range(len(W_new)):
    if W_new[i].shape == W_old[i].shape:
        W_new[i] = W_old[i]

# dump new weights to the model
model.set_weights(W_new)
print('... done')

VARs = np.empty(shape_3km+(N_var,))
VARs[...] = np.nan

FEATURE_VEC = np.empty(shape_80km+(N_lead, L_vec))
FEATURE_VEC[...] = np.nan

input_frame = np.empty((1, input_size, input_size, N_var))
input_frame[...] = np.nan

PROB = np.empty(shape_80km+(N_lead,))
PROB[...] = np.nan

print("Converting HRRR 3-km field into feature vectors")

for l in range(N_lead):
    lead = leads[l]
    print('Pre-rpocessing {}-hr forecasts ...'.format(lead))
    start_time = time.time()
    
    filename_grib = (datetime.strftime(date_temp, HRRR_dir+HRRR_name)).format(lead, lead)

    var_names_temp = []
    with pygrib.open(filename_grib) as grbio:
        for i, ind in enumerate(HRRRv4_inds):
            var_names_temp.append(str(grbio[ind])[:35])

    flag_qc = var_names == var_names_temp
    print("HRRR quality control flag = {}".format(flag_qc))
    
    with pygrib.open(filename_grib) as grbio:
        for i, ind in enumerate(HRRRv4_inds):
            VARs[..., i] = grbio[ind].values
        
    for ix in range(shape_80km[0]):
        for iy in range(shape_80km[1]):

            indx = int(indx_array[ix, iy])
            indy = int(indy_array[ix, iy])

            x_edge_left = indx - half_margin
            x_edge_right = indx + half_margin

            y_edge_bottom = indy - half_margin
            y_edge_top = indy + half_margin

            if x_edge_left >= 0 and y_edge_bottom >= 0 and x_edge_right < shape_3km[0] and y_edge_top < shape_3km[1]:

                hrrr_temp = VARs[x_edge_left:x_edge_right, y_edge_bottom:y_edge_top, :]

                for n in range(N_var):

                    means = mean_stats[ix, iy, n, l]
                    stds = std_stats[ix, iy, n, l]
                    max_vals = max_stats[ix, iy, n, l]

                    temp = hrrr_temp[..., n]

                    # (n==0) Radar reflectivity, correct negative to 0
                    if n == 0:
                        temp[temp<0] = 0

                    # (n==10) CIN, preserve negative vals only, and convert them to positive 
                    if n == 10:
                        temp = -1*temp
                        temp[temp<0] = 0

                    # variables that will be normalizaed with log transformation
                    if log_norm[n]:
                        temp = np.log(np.abs(temp)+1)
                        # for CIN and SRH, x3 the value
                        if n < 9:
                            temp = temp/stds/max_vals
                        else:
                            temp = 3.0*temp/stds/max_vals

                    else:
                        temp = (temp - means)/stds

                    input_frame[..., n] = temp

                # CNN feature vectors

                temp_vec = model.predict([input_frame])
                FEATURE_VEC[ix, iy, l, :] = temp_vec[0, :]
                
    print("--- %s seconds ---" % (time.time() - start_time))
    print('...done')
    
for l in range(N_lead-1):
    
    lead = leads[l]
    print('Estimating probabilities on {}-hr forecasts ...'.format(lead))
    start_time = time.time()
    
    if l == 0:
        N_vec = 2*2
        lead_range = [0, 1]

    elif l == 1:
        N_vec = 3*2
        lead_range = [0, 1, 2]

    else:
        N_vec = 4*2
        lead_range = [l-2, l-1, l, l+1]
    
    # ======================================= #
    # load classifier head
    model_head = mu.create_model_head(N_vec, L_vec)
    #W_new = model.get_weights()
    W_old = mu.dummy_loader(model_head_name.format(lead))
    model_head.set_weights(W_old)
    # ======================================= #
    
    VEC_merge = np.empty((N_vec, L_vec))
    VEC_merge[...] = np.nan
    
    for ix in range(shape_80km[0]):
        for iy in range(shape_80km[1]):
            
            count = 0
            #vec_merge = ()

            indx_temp = ix
            indy_temp = iy

            indx_left = np.max([indx_temp - 1, 0])
            indx_right = np.min([indx_temp + 1, shape_80km[0]-1])

            indy_bot = np.max([indy_temp - 1, 0])
            indy_top = np.min([indy_temp + 1, shape_80km[1]-1])
                
            for ix_vec in [indx_temp, indx_left, indx_right]:
                for iy_vec in [indy_temp, indy_bot, indy_top]:
                    for il in lead_range:
                        vec_temp = FEATURE_VEC[ix_vec, iy_vec, il, :]
                        if np.sum(np.isnan(vec_temp)) == 0 and count < N_vec:
                            #vec_merge += (vec_temp[None, ...],)
                            VEC_merge[count, :] = vec_temp
                            count += 1
                                
            if count < N_vec:
                continue
            else:
                #VEC_merge = np.concatenate(vec_merge, axis=0)
                Input_vec = VEC_merge[None, ...]


                lon = lon_80km[ix, iy]
                lat = lat_80km[ix, iy]

                lon = (lon - lon_minmax[0])/(lon_minmax[1] - lon_minmax[0])
                lat = (lat - lat_minmax[0])/(lat_minmax[1] - lat_minmax[0])
                Input_stn = np.array([lon, lat])
                Input_stn = Input_stn[None, ...]

                prob_raw = model_head.predict([Input_vec, Input_stn])
                PROB[ix, iy, l] = prob_raw[0]
                
    print("--- %s seconds ---" % (time.time() - start_time))
    print('... done')
    
tuple_save = (FEATURE_VEC, PROB)
label_save = ['FEATURE_VEC', 'PROB']
du.save_hdf5(tuple_save, label_save, output_dir, datetime.strftime(date_temp, output_name))
