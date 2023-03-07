
# input size, e.g., 64-by-64
input_size = 64

# length of feature vectors
L_vec = 128

# forcast lead times
leads = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Location of HRRR v4 grib
HRRR_dir = '/glade/campaign/cisl/aiml/ksha/HRRR/'

# file sub-folder and name
# test the following to make sure it is right:
# (datetime.strftime(date_temp, HRRR_dir+'fcst{:02d}hr/HRRR.%Y%m%d.natf{:02d}.grib2')).format(lead, lead)
HRRR_name = 'fcst{:02d}hr/HRRR.%Y%m%d.natf{:02d}.grib2'

# Location of domain and stats files
save_dir = '/glade/work/ksha/NCAR/'

# location of output file
output_dir = '/glade/work/ksha/NCAR/'

# output name
output_name = 'output_%Y%m%d.hdf'

# CNN weights
model_name = '/glade/work/ksha/NCAR/Keras_models/RE2_peak_base5/'

# Classifier head weights
model_head_name = '/glade/work/ksha/NCAR/Keras_models/peak_lead{}/'

# ========================== #
# Grib file section
# Variable indices are difined based on NAT* files
# ========================== #

# Grib indices in HRRR v4 file
HRRRv4_inds = [1003, 1014, 1018, 1020, 1028, 1041, 1044, 1049, 1060, 1074, 1075, 1097, 1098, 1103, 1104]
# ['Max/Comp Radar', 'MSLP', 'UH 2-5 km', 'UH 0-2 km', 'Graupel mass', 'T 2m', 'Dewpoint 2m', 'SPD 10m',
#  'APCP', 'CAPE', 'CIN', 'SRH 0-3 km', 'SRH 0-1 km', 'U shear 0-6 km', 'V shear 0-6 km']

# variable name patterns will be checked to avoid pulling the woring variables.
var_names = ['1003:Maximum/Composite radar reflec',
             '1014:MSLP (MAPS System Reduction):P',
             '1018:199:199 (max):lambert:heightAb',
             '1020:199:199 (max):lambert:heightAb',
             '1028:74:74 (max):lambert:atmosphere',
             '1041:2 metre temperature:K (instant',
             '1044:2 metre dewpoint temperature:K',
             '1049:10 metre wind speed:m s**-1 (m',
             '1060:Total Precipitation:kg m**-2 (',
             '1074:Convective available potential',
             '1075:Convective inhibition:J kg**-1',
             '1097:Storm relative helicity:J kg**',
             '1098:Storm relative helicity:J kg**',
             '1103:Vertical u-component shear:s**',
             '1104:Vertical v-component shear:s**']

# True: noramlize by log-transformation
# False: normalize by standardization
log_norm = [True, False, True, True, True, False, False, True, True, True, True, False, False, False, False]






