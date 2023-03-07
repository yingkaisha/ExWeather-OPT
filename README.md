# ExWeather-OPT
Operational severe weather prediction system

# Dependency

* h5py==2.10.0
* numpy==1.20.3
* pygrib==2.1.4
* scipy==1.9.3
* tensorflow==2.4.1

# Usage
* Edit namelist.py
* `python main.py 2021 2 1` e.g., for 2021 Feb 1st
* output: a hdf5 file that contains feature vectors and prabilities
```python
import h5py

with h5py.File('/glade/work/ksha/NCAR/output_20210201.hdf', 'r') as h5io:
    feature_vector = h5io['FEATURE_VEC'][...]
    severe_weather = h5io['PROB'][...]
```

