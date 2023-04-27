#!/usr/bin/python3
import sys
import re


from . import zoidberg_examples as zbe
from boututils.datafile import DataFile
import zoidberg
from . import version, soft_name

# conf          name
# 0	w7x	standard case
# 3	w7x	low iota
# 4	w7x	high iota
# 6	w7x	high mirror


debug = False


def gen(conf, shape, fn, **kw):
    print(fn)
    with zbe.timeit():
        zbe.W7X(
            *shape,
            configuration=conf,
            calc_curvature=True,
            show_lines=debug,
            show_maps=debug,
            fname=fn,
            **kw,
        )
        with DataFile(fn, write=True) as f:
            f.write_file_attribute(soft_name + "_version", version)
            f.write_file_attribute("zoidberg_version", zoidberg.__version__)
            f.write_file_attribute("software_name", soft_name)
            f.write_file_attribute("software_version", version)


def gen2(fn):
    args = re.match(r"^W7X-conf(\d+)-(\d+)x(\d+)x(\d+)(.*).fci(\d*).nc$", fn)
    if not args:
        print(f"'{fn}' is not a valid filename")
        return

    args = args.groups()
    args = [int(x) for x in args[:4]] + list(args[4:])
    conf = args[0]
    shape = args[1:4]
    known_opts = dict(
        full="full_torus",
        outer="outer_VMEC",
        island="outer_islands",
        vessel="outer_vessel",
        inner="inner_VMEC",
        emc3="emc3_file",
    )
    kw = dict(
        outer_VMEC=False,
        outer_islands=True,
        outer_vessel=True,
        inner_VMEC=True,
    )
    print(args[4:])
    for opt in args[4].split("."):
        print(opt)
        if not opt:
            continue
        if "=" in opt or ":" in opt:
            if "=" in opt:
                opt, val = opt.split("=")
            else:
                opt, val = opt.split(":")

            try:
                val = float(val)
            except ValueError:
                try:
                    val = {
                        "True": True,
                        "T": True,
                        "t": True,
                        "False": False,
                        "F": False,
                        "f": False,
                    }[val]
                except KeyError:
                    pass
            else:
                if abs(val - int(val)) < 1e-6:
                    val = int(val)
        else:
            val = True
        kw[known_opts[opt]] = val
    print(kw)
    gen(conf, shape, fn, field_refine=0, **kw)


def help(exit=0):
    print(f"Usage: {sys.argv[0]} <filename>")
    print("Where <filename> is of the format W7X-conf<conf>-<nx>x<ny>x<nz>.fci")
    print(" * with <conf> the configuration id,")
    print(" * with <n?> the number of grid points in that direction")
    sys.exit(exit)


def main():
    global debug
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
