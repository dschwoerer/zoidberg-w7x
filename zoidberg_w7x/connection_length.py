#!/usr/bin/python3
import sys
import re


fn = sys.argv[1]
yid = int(sys.argv[2])

args = re.findall(r"W7X-conf(\d+)-(\d+)x(\d+)x(\d+)(.*).fci", fn)
if not args:
    raise ValueError(f"'{fn}' is not a valid filename")

(args,) = args
print(args)
conf = args[0]

try:
    import proxy_local_12345
except ImportError:
    print("whaa")
    raise
    pass

import xarray as xr

with xr.open_dataset(fn) as ds:
    phi = ds.phi.rename(t="y")
    ds["phi"] = phi
    ds = ds.isel(y=yid)
    print(ds)
    print(list(ds))
    print(ds.phi)
    from zoidberg_examples import connection_length

    connection_length((ds.R, ds.Z), ds.phi, conf, 100)
