# ExWeather-OPT
Operational severe weather prediction system

Usage:
* Edit namelist.py
* `python main.py 2021 2 1` e.g., for 2021 Feb 1st
* output: a hdf5 file that contains feature vectors and prabilities
```python
import h5py

with h5py.File('/glade/work/ksha/NCAR/output_20210201.hdf', 'r') as h5io:
    feature_vector = h5io['FEATURE_VEC'][...]
    severe_weather = h5io['PROB'][...]
```

