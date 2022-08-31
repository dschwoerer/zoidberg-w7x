#!/usr/bin/python3
import sys
import re

from . import zoidberg_examples as zbe
from boututils.datafile import DataFile

# conf          name
# 0	w7x	standard case
# 3	w7x	low iota
# 4	w7x	high iota
# 6	w7x	high mirror


try:
    import proxy_local_12345
except ImportError:
    pass

debug = False


def gen(conf, shape, **kw):
    fn = f"""W7X-conf{conf}-{"x".join([str(d) for d in shape])}.fci.nc"""
    print(fn)
    with zbe.timeit():
        zbe.W7X(
            *shape,
            outer_vessel=True,
            inner_VMEC=True,
            outer_VMEC=True,
            configuration=conf,
            calc_curvature=True,
            show_lines=debug,
            show_maps=debug,
            fname=fn,
            **kw,
        )
        with DataFile(fn, write=True) as f:
            for ff in __file__, zbe.__file__:
                with open(ff) as f2:
                    code = f2.read()
                ff = ff.split("/")[-1]
                f.write_file_attribute(ff, code)


def gen2(fn):
    args = re.findall(r"W7X-conf(\d+)-(\d+)x(\d+)x(\d+).fci", fn)
    if not args:
        print(f"'{fn}' is not a valid filename")
        return

    assert len(args) == 1
    args = [int(x) for x in args[0]]
    conf = args[0]
    shape = args[1:]
    gen(conf, shape, field_refine=0)


def help(exit=0):
    print(f"Usage: {sys.argv[0]} <filename>")
    print("Where <filename> is of the format W7X-conf<conf>-<nx>x<ny>x<nz>.fci")
    print(" * with <conf> the configuration id,")
    print(" * with <n?> the number of grid points in that direction")
    sys.exit(exit)


def main():
    opts = {k: True for k in sys.argv[1:]}
    debug = opts.pop("--debug", False)
    if "--help" in opts:
        help()
    if not opts:
        help(1)
    for fn in opts:
        gen2(fn)


if __name__ == "__main__":
    main()
