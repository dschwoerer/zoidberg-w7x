#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import random
import time
from functools import partial
from multiprocessing import Pool

import boututils.calculus as calc
import matplotlib.pyplot as plt
import numpy as np
import zoidberg as zb
from boututils.datafile import DataFile


def screwpinch(
    nx=68,
    ny=16,
    nz=128,
    xcentre=1.5,
    fname="screwpinch.fci.nc",
    a=0.2,
    npoints=421,
    show_maps=False,
):
    yperiod = 2 * np.pi
    field = zb.field.Screwpinch(xcentre=xcentre, yperiod=yperiod, shear=0 * 2e-1)
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre + a / 2.0
    start_z = 0.0
    print("Making curvilinear poloidal grid")
    inner = zb.rzline.shaped_line(
        R0=xcentre, a=a / 2.0, elong=0, triang=0.0, indent=0, n=npoints
    )
    outer = zb.rzline.shaped_line(
        R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints
    )
    poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz)
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps = zb.make_maps(grid, field)
    zb.write_maps(grid, field, maps, str(fname), metric2d=False)
    calc_curvilinear_curvature(fname, field, grid, maps)

    if show_maps:
        zb.plot.plot_forward_map(grid, maps, yslice=0)


def rotating_ellipse(
    nx=68,
    ny=16,
    nz=128,
    xcentre=5.5,
    I_coil=0.01,
    curvilinear=True,
    rectangular=False,
    fname="rotating-ellipse.fci.nc",
    a=0.4,
    curvilinear_inner_aligned=True,
    curvilinear_outer_aligned=True,
    curvilinear_outer_limited=False,
    npoints=421,
    Btor=2.5,
    show_maps=False,
    calc_curvature=True,
    smooth_curvature=False,
    return_iota=True,
    write_iota=False,
):
    yperiod = 2 * np.pi / 5.0
    field = zb.field.RotatingEllipse(
        xcentre=xcentre, I_coil=I_coil, radius=2 * a, yperiod=yperiod, Btor=Btor
    )
    # Define the y locations
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre + a / 2.0
    start_z = 0.0

    if rectangular:
        print("Making rectangular poloidal grid")
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(
            nx, nz, 1.0, 1.0, Rcentre=xcentre
        )
    elif curvilinear:
        print("Making curvilinear poloidal grid")
        inner = zb.rzline.shaped_line(
            R0=xcentre, a=a / 2.0, elong=0, triang=0.0, indent=0, n=npoints
        )
        outer = zb.rzline.shaped_line(
            R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints
        )

        if curvilinear_inner_aligned:
            print("Aligning to inner flux surface...")
            inner_lines = get_lines(
                field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )
        if curvilinear_outer_aligned:
            print("Aligning to outer flux surface...")
            outer_lines = get_lines(
                field, xcentre + a, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )
        if curvilinear_outer_limited:
            for outer in outer_lines:
                print(
                    np.min(outer.Z), np.max(outer.Z), np.min(outer.R), np.max(outer.R)
                )
            raise

        print("creating grid...")
        if curvilinear_inner_aligned:
            if curvilinear_outer_aligned:
                poloidal_grid = [
                    zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps)
                    for inner, outer in zip(inner_lines, outer_lines)
                ]
            else:
                poloidal_grid = [
                    zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps)
                    for inner in inner_lines
                ]
        else:
            poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz)

    # Create the 3D grid by putting together 2D poloidal grids
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps = zb.make_maps(grid, field)
    zb.write_maps(grid, field, maps, str(fname), metric2d=False)

    if curvilinear and calc_curvature:
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid, maps)

    if calc_curvature and smooth_curvature:
        smooth_metric(
            fname, write_to_file=True, return_values=False, smooth_metric=True
        )

    if return_iota or write_iota:
        rindices = np.linspace(start_r, xcentre + a, nx)
        zindices = np.zeros((nx))
        iota_bar = calc_iota(field, start_r, start_z)
        if write_iota:
            f = DataFile(str(fname), write=True)
            f.write("iota_bar", iota_bar)
            f.close()
        else:
            print("Iota_bar = ", iota_bar)


def dommaschk(
    nx=68,
    ny=16,
    nz=128,
    C=None,
    xcentre=1.0,
    Btor=1.0,
    a=0.1,
    curvilinear=True,
    rectangular=False,
    fname="Dommaschk.fci.nc",
    curvilinear_inner_aligned=True,
    curvilinear_outer_aligned=True,
    npoints=421,
    show_maps=False,
    calc_curvature=True,
    smooth_curvature=False,
    return_iota=True,
    write_iota=False,
):

    if C is None:
        C = np.zeros((6, 5, 4))
        C[5, 2, 1] = 0.4
        C[5, 2, 2] = 0.4
        # C[5,4,1] = 19.25

    yperiod = 2 * np.pi / 5.0
    field = zb.field.DommaschkPotentials(C, R_0=xcentre, B_0=Btor)
    # Define the y locations
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    start_r = xcentre + a / 2.0
    start_z = 0.0

    if rectangular:
        print("Making rectangular poloidal grid")
        poloidal_grid = zb.poloidal_grid.RectangularPoloidalGrid(
            nx, nz, 1.0, 1.0, Rcentre=xcentre
        )
    elif curvilinear:
        print("Making curvilinear poloidal grid")
        inner = zb.rzline.shaped_line(
            R0=xcentre, a=a / 2.0, elong=0, triang=0.0, indent=0, n=npoints
        )
        outer = zb.rzline.shaped_line(
            R0=xcentre, a=a, elong=0, triang=0.0, indent=0, n=npoints
        )

        if curvilinear_inner_aligned:
            print("Aligning to inner flux surface...")
            inner_lines = get_lines(
                field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )
        if curvilinear_outer_aligned:
            print("Aligning to outer flux surface...")
            outer_lines = get_lines(
                field, xcentre + a, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )

        print("creating grid...")
        if curvilinear_inner_aligned:
            if curvilinear_outer_aligned:
                poloidal_grid = [
                    zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps)
                    for inner, outer in zip(inner_lines, outer_lines)
                ]
            else:
                poloidal_grid = [
                    zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz, show=show_maps)
                    for inner in inner_lines
                ]
        else:
            poloidal_grid = zb.poloidal_grid.grid_elliptic(inner, outer, nx, nz)

    # Create the 3D grid by putting together 2D poloidal grids
    grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)
    maps = zb.make_maps(grid, field)
    zb.write_maps(grid, field, maps, str(fname), metric2d=False)

    if curvilinear and calc_curvature:
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid, maps)

    if calc_curvature and smooth_curvature:
        smooth_metric(
            fname, write_to_file=True, return_values=False, smooth_metric=False
        )

    if return_iota or write_iota:
        rindices = np.linspace(start_r, xcentre + a, nx)
        zindices = np.zeros((nx))
        iota_bar = calc_iota(field, start_r, start_z)
        if write_iota:
            f = DataFile(str(fname), write=True)
            f.write("iota_bar", iota_bar)
            f.close()
        else:
            print("Iota_bar = ", iota_bar)


def W7X(
    nx=68,
    ny=32,
    nz=256,
    fname="W7-X.fci.nc",
    vmec_file="w7-x.wout.nc",
    inner_VMEC=False,
    inner_vacuum=False,
    outer_VMEC=False,
    outer_vacuum=False,
    outer_vessel=False,
    outer_poincare=False,
    npoints=100,
    a=2.5,
    show_maps=False,
    show_lines=False,
    calc_curvature=True,
    smooth_curvature=False,
    plasma_field=False,
    configuration=0,
    vmec_url=None,
    field_refine=1,
    trace_web=True,
    avoid_double_outer=True,
    full_torus=False,
):

    if vmec_url is None:
        urls = {
            0: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/w7x_ref_171/wout.nc",
            3: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0750_+0750/01/00/wout.nc",
            4: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_-0690_-0690/01/00/wout.nc",
        }
        if configuration not in urls:
            raise KeyError(
                "Do not know the appropriate vmec url for that configuartion.\n"
                "Check http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/"
            )
        vmec_url = urls[configuration]

    yperiod = 2 * np.pi if full_torus else 2 * np.pi / 5.0
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)
    outer_lines = None
    history_outer = {}
    if outer_VMEC:
        print("Aligning to outer VMEC flux surface...")
        if isinstance(outer_VMEC, bool):
            outer_VMEC = 1.5
        outer_new = get_VMEC_surfaces(
            phi=ycoords, s=outer_VMEC, npoints=nz * 10, w7x_run=vmec_url
        )
        outer_lines = get_inner(outer_new, outer_lines)
        if show_lines:
            history_outer["vmec"] = outer_new
    if outer_vessel:
        print("Aligning to plasma vessel (EXPERIMENTAL) ...")
        with timeit(" ... took %f"):
            outer_new = get_W7X_vessel(phi=ycoords, nz=nz * 10)
            outer_lines = get_inner(outer_new, outer_lines)
            assert_valid(outer_lines)
        if show_lines:
            history_outer["vessel"] = outer_new

    if outer_poincare:
        R0 = {0: 6.3, 4: 6.17}[configuration]
        print(f"Aligning to poincare R={R0}")
        with timeit(" ... took %f"):
            outer_new = get_W7X_poincare(
                R0, phi=ycoords, conf=configuration, nz=nz * 10
            )
            outer_lines = get_inner(outer_new, outer_lines)
        if show_lines:
            history_outer["poincare"] = outer_new

    elif outer_vacuum or inner_vacuum:
        xmin = 4.05
        xmax = 4.05 + 2.5
        zmin = -1.35
        zmax = -zmin
        field = zb.field.W7X_vacuum(
            phimax=yperiod,
            x_range=[xmin, xmax],
            z_range=[zmin, zmax],
            include_plasma_field=plasma_field,
        )
        xcentre, zcentre = field.magnetic_axis(
            phi_axis=ycoords[0], configuration=configuration
        )
        start_r = xcentre + a / 2.0
        start_z = zcentre
        if inner_vacuum:
            print("Aligning to inner vacuum flux surface...")
            inner_lines = get_lines(
                field, start_r, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )
        if outer_vacuum:
            print("Aligning to outer vacuum flux surface...")
            outer_lines = get_lines(
                field, start_r + a, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )

    xmin = np.min([min(outer_lines[i].R) for i in range(ny)])
    xmax = np.max([max(outer_lines[i].R) for i in range(ny)])
    zmin = np.min([min(outer_lines[i].Z) for i in range(ny)])
    zmax = np.max([max(outer_lines[i].Z) for i in range(ny)])

    with timeit("Creating a field took %f"):
        if field_refine:
            field = zb.field.W7X_vacuum(
                *[x * field_refine for x in (128, 32, 128)],
                phimax=yperiod,
                x_range=[xmin, xmax],
                z_range=[zmin, zmax],
                include_plasma_field=plasma_field,
                configuration=configuration,
            )
        else:
            field = zb.field.W7X_vacuum_on_demand(configuration)

    if inner_VMEC:
        print("Aligning to inner VMEC flux surface...")
        with timeit(" ... took %f"):
            inner_lines = get_VMEC_surfaces(
                phi=ycoords, s=0.67, npoints=nz * 2, w7x_run=vmec_url
            )

    tracer = (
        zb.fieldtracer.FieldTracerWeb(configId=configuration, stepsize=0.01)
        if trace_web
        else None
    )
    if avoid_double_outer:
        print("Avoid double boundary")

        def trace(outer_lines):
            m = hashlib.md5()
            m.update("-".join([str(x) for x in ycoords]).encode())
            fn0 = fn = f"w7x_vessel_{nz}_{m.hexdigest()[:10]}"
            fn = f"{fn0}.cache"
            try:
                dat = np.loadtxt(fn)
                dat.shape = (2, -1, 2, dat.shape[-1])
                return dat
            except:
                print("nocache:", fn)

            dy = ycoords[1] - ycoords[0]
            dat = [], []
            with Pool(16) as pool:
                tracerfunc = partial(
                    tracer.follow_field_lines, chunk=int(1e8), timeout=60, retry=2
                )
                for outer, y in zip(outer_lines, ycoords):
                    for lst, dist in zip(dat, [dy, -dy]):
                        lst += [
                            pool.apply_async(
                                tracerfunc, (outer.R, outer.Z, [y, y + dist])
                            )
                        ]
                k = 0
                with zb.progress.Progress() as prog:
                    prog.update(0)
                    for da in dat:
                        for i in range(len(da)):
                            da[i] = da[i].get()
                            k += 1
                            prog.update(k / len(da) / 2)
            for da in dat:
                for i in range(len(da)):
                    da[i] = np.transpose(da[i][1])
            dat = np.array(dat)
            shape = dat.shape
            dat.shape = (-1, shape[-1])
            np.savetxt(fn, dat)
            dat.shape = shape
            return dat

        fwd, bwd = trace(outer_lines)
        fwd = np.roll(fwd, 1, axis=0)
        bwd = np.roll(bwd, -1, axis=0)
        union = get_outer(fwd, bwd)
        history_outer["avoid_double"] = union
        outer_lines = get_inner(outer_lines, union)

    if show_lines:
        for i in range(ny):
            plt.figure()
            plt.plot(
                *inner_lines[i].position(np.linspace(0, 2 * np.pi, 10 * nz)),
                label="inner",
            )
            for k, v in history_outer.items():
                plt.plot(
                    *v[i].position(np.linspace(0, 2 * np.pi, 10 * nz)),
                    label=f"outer_{k}",
                )
            plt.plot(
                *outer_lines[i].position(np.linspace(0, 2 * np.pi, 10 * nz)),
                label="outer",
            )
            plt.legend()
            plt.title(f"{i} phi={ycoords[i]}")
        plt.show()

    print("creating grid...")
    with timeit("Creating poloidal grids took %f"):
        poloidal_grid = [
            zb.poloidal_grid.grid_elliptic(
                inner, outer, nx, nz, show=show_maps, nx_outer=2
            )
            for inner, outer in zip(inner_lines, outer_lines)
        ]

    # Create the 3D grid by putting together 2D poloidal grids
    with timeit("Creating a grid took %f"):
        grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)

    with timeit("Creating maps took %f"):
        maps = zb.make_maps(grid, field, field_tracer=tracer)
    maps["phi"] = ycoords
    zb.write_maps(grid, field, maps, str(fname), metric2d=False)

    if calc_curvature:
        print("calculating curvature...")
        calc_curvilinear_curvature(fname, field, grid, maps)

    if calc_curvature and smooth_curvature:
        smooth_metric(
            fname, write_to_file=True, return_values=False, smooth_metric=True
        )

    if show_maps:
        zb.plot.plot_forward_map(grid, maps, yslice=-1)


def get_inner(lines1, lines2):
    return get_inner_outer(lines1, lines2, True)


def get_outer(lines1, lines2):
    return get_inner_outer(lines1, lines2, False)


def assert_valid(line):
    if isinstance(line, list):
        for l in line:
            assert_valid(l)
        return
    if isinstance(line, np.ndarray):
        if len(line.shape) > 2:
            for l in line:
                assert_valid(l)
            return
        if line.shape[0] == 2:
            x, y = line
        else:
            x, y = line.T
    else:
        x = line.R
        y = line.Z
    dx = x - np.roll(x, 1)
    dy = y - np.roll(y, 1)
    dr = dx**2 + dy**2
    bad = dr == 0
    if not np.any(bad):
        return
    print(np.sum(bad))
    print(np.arange(len(bad))[bad])
    plt.plot(x - 5)
    plt.plot(y)
    plt.plot(dx * 1000)
    plt.plot(dy * 1000)
    plt.figure()
    plt.plot(x, y, "x-")
    plt.show()

    assert 0


def get_inner_outer(lines1, lines2, io):
    if lines2 is None:
        return lines1
    import shapely.geometry as sg

    assert_valid(lines1)
    assert_valid(lines2)

    out = []
    for l1, l2 in zip(lines1, lines2):
        if isinstance(l1, zb.rzline.RZline):
            l1 = np.array([l1.R, l1.Z]).T
        else:
            l1 = l1.T
        if isinstance(l2, zb.rzline.RZline):
            l2 = np.array([l2.R, l2.Z]).T
        else:
            l2 = l2.T
        nz = len(l1)
        if nz != len(l2):
            nz = None

        ps = []
        for li in l1, l2:
            try:
                pi = sg.Polygon(li)
                assert pi.is_valid
            except:
                plt.plot(*li.T, "x-")
                plt.figure()
                plt.plot(li.T[0], "x-")
                plt.plot(li.T[1], "o-")
                x, y = li.T
                dx = x - np.roll(x, 1)
                dy = y - np.roll(y, 1)
                plt.figure()
                plt.scatter(dx, dy)
                plt.show()
                raise
            ps.append(pi)
        p1, p2 = ps

        try:
            if io:
                res = p1.intersection(p2)
            else:
                res = p1.union(p2)
            if isinstance(res, sg.MultiPolygon):
                plt.plot(*l1.T)
                plt.plot(*l2.T)
                res = list(res)
                best = None
                for r in res:
                    plt.plot(*np.array(r.exterior.coords).T, label=r.area)
                    if best is None or r.area > best.area:
                        best = r
                plt.legend()
                plt.show()
                res = best
            newrz = res.exterior.coords
        except:
            plt.plot(*l1.T, "x-", label="l1")
            plt.plot(*l2.T, "x-", label="l2")
            plt.legend()
            plt.show()
            raise

        line = zb.rzline.line_from_points(
            *np.array(newrz).T, spline_order=1, is_sorted=True
        )
        if nz:
            line = line.equallySpaced(n=nz)
        out.append(line)
    return out


def get_tracer(_cache={}):
    from osa import Client

    if "tracer" not in _cache:
        _cache["tracer"] = Client("http://esb:8280/services/FieldLineProxy?wsdl")
    return _cache["tracer"]


def connection_length(RZ, phi, conf, limit):
    m = hashlib.md5()
    for var in [*RZ, phi]:
        m.update("-".join([str(x) for x in var]).encode())
    m.update("-".join([str(x) for x in phi]).encode())
    fn = f"w7x_poincare_{conf}_{m.hexdigest()[:10]}.cache"
    try:
        dat = np.loadtxt(fn)
        dat.shape = (2, -1, 2, dat.shape[-1])
        return dat
    except:
        print("nocache:", fn)
    flt = get_tracer()
    pnts = flt.types.Points3D()


def trace_boundary(fn, conf):
    with DataFile(fn) as dat:
        nx = dat["nx"]
        mxg = dat.get("MXG", 2)
        myg = dat.get("MYG", 1)
        phi = dat["phi"]
        limit = np.max(dat["R"]) * np.max(dat["dy"]) * (myg + 0.5)

        for fwd, dir in zip(("forward", "backward"), (1, -1)):
            xtp = dat[fwd + "_xt_prime"]
            bnd = xtp > nx - mxg - 1
            RZ = dat["R"][bnd], dat["Z"][bnd]
        return


def trace_poincare(start_r, start_z, yslices, npoints, conf, symmetry=5, step=1e-3):
    import xarray as xr

    m = hashlib.md5()
    start_r = np.array(start_r)
    start_z = np.array(start_z)
    for var in start_r, start_z, yslices, [symmetry, npoints, step]:
        m.update("-".join([str(x) for x in var]).encode())

    assert yslices[0] == 0
    fn = f"w7x_poincare_{conf}_{m.hexdigest()[:10]}.cache.nc"
    try:
        dat = xr.open_dataarray(fn)
        return dat
    except:
        print("nocache:", fn)

    yslices0 = np.array(yslices).copy()
    if symmetry > 1:
        ylen = len(yslices)
        yslices = np.empty(ylen * symmetry)
        offsets = np.linspace(0, 2 * np.pi, symmetry, endpoint=False)
        for i, y0 in enumerate(yslices0):
            yslices[i * symmetry : (i + 1) * symmetry] = y0 + offsets

    flt = get_tracer()
    pnts = flt.types.Points3D()
    pnts.x1 = start_r * np.cos(yslices[0])
    pnts.x2 = start_r * np.sin(yslices[0])
    pnts.x3 = start_z
    # print(pnts)

    config = flt.types.MagneticConfig()
    config.configIds = [conf]

    poincare = flt.types.PoincareInPhiPlane()
    poincare.numPoints = npoints
    poincare.phi0 = yslices

    task = flt.types.Task()
    task.step = step
    task.poincare = poincare

    dat = np.empty((2, len(yslices0), 2, symmetry, len(pnts.x1), npoints)) * np.nan
    for i3 in 0, 1:
        config.inverseField = bool(i3)

        res = flt.service.trace(pnts, config, task, None, None)

        i = 0
        for i0 in range(len(pnts.x1)):
            for i1 in range(len(yslices0)):
                for i2 in range(symmetry):
                    s = res.surfs[i]
                    xyz = np.array([s.points.x1, s.points.x2, s.points.x3])
                    r = np.sqrt(np.sum(xyz[:2] ** 2, axis=0))
                    ln = len(r)
                    if ln != npoints:
                        print(i0, i1, i2, i3, ln, npoints)
                    assert np.isclose(yslices0[i1], s.phi0 % (2 * np.pi / symmetry))
                    dat[0, i1, i3, i2, i0, :ln] = r
                    dat[1, i1, i3, i2, i0, :ln] = xyz[2]
                    i += 1
    da = xr.DataArray(
        dat,
        dims=("Rz", "phi", "direction", "symmetry", "index", "point"),
        coords=dict(
            Rz=["R", "z"], phi=np.array(yslices0), direction=["forward", "backward"]
        ),
    )
    da.attrs = dict(step=step, configuration=conf)
    da.to_netcdf(fn)
    return da


def get_W7X_poincare(R, phi, conf, nz):
    dat = _get_W7X_poincare(R, phi, conf)
    out = []
    for cur in dat.transpose("phi", ...):
        cur = zb.rzline.RZline(*cur, smooth=50)
        cur = cur.equallySpaced(n=nz)
        out.append(cur)
    return out


def _get_W7X_poincare(R, phi, conf):
    import xarray as xr

    m = hashlib.md5()
    m.update("-".join([str(x) for x in phi]).encode())

    fn = f"w7x_poincare_{conf}_{R}_{m.hexdigest()[:10]}.cache.nc"
    try:
        return xr.open_dataarray(fn)
    except FileNotFoundError:
        pass

    pnc = trace_poincare([R], [0], phi, 50, conf)

    def _prod(a):
        r = 1
        for k in a:
            r *= k
        return r

    newshape = (pnc.shape[0], pnc.shape[1], _prod(pnc.shape[2:]))
    out = np.empty(newshape)
    for i, phi in enumerate(pnc.phi):
        rz = pnc.sel(phi=phi)
        cur = zb.rzline.line_from_points(*[k.data.flatten() for k in rz], smooth=50)
        out[:, i, :] = np.array([cur.R, cur.Z])

    da = xr.DataArray(
        out, dims=(*pnc.dims[:2], "points"), coords=dict(Rz=pnc["Rz"], phi=pnc["phi"])
    )
    da.to_netcdf(fn)
    return out


def plot_poincare(*args, **kw):
    da = trace_poincare(*args, **kw)
    for phi in da.phi:
        dap = da.sel(phi=phi)
        plt.figure()
        for index in dap.index:
            plt.scatter(*dap.sel(index=index))
        plt.title(f"phi = {phi.data}")

    plt.show()


def get_lines(
    field, start_r, start_z, yslices, yperiod=2 * np.pi, npoints=150, smoothing=False
):
    rzcoord, ycoords = zb.fieldtracer.trace_poincare(
        field, start_r, start_z, yperiod, y_slices=yslices, revs=npoints
    )

    lines = []
    for i in range(ycoords.shape[0]):
        r = rzcoord[:, i, 0, 0]
        z = rzcoord[:, i, 0, 1]
        line = zb.rzline.line_from_points(r, z)
        # Re-map the points so they're approximately uniform in distance along the surface
        # Note that this results in some motion of the line
        line = line.equallySpaced()
        lines.append(line)

    return lines


### return a VMEC flux surface as a RZline object
def get_VMEC_surfaces(phi=[0], s=0.75, w7x_run="w7x_ref_1", npoints=100):
    from osa import Client

    client = Client("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl")

    if s > 1:
        assert s < 2
        points1 = client.service.getFluxSurfaces(str(w7x_run), phi, 1, npoints)
        pointss = client.service.getFluxSurfaces(str(w7x_run), phi, 2 - s, npoints)
    else:
        points = client.service.getFluxSurfaces(str(w7x_run), phi, s, npoints)

    lines = []
    for y in range(len(phi)):
        if s > 1:
            rz1 = np.array((points1[y].x1, points1[y].x3))
            rzs = np.array((pointss[y].x1, pointss[y].x3))
            rz = rz1 * 2 - rzs
        else:
            rz = points[y].x1, points[y].x3
        line = zb.rzline.RZline(*rz)
        line = line.equallySpaced()
        lines.append(line)

    return lines


### Return the W7X PFC as RZline objects
def get_W7X_vessel(phi=[0], nz=256, show=False):
    m = hashlib.md5()
    m.update("-".join([str(x) for x in phi]).encode())
    fn = f"w7x_vessel_{nz}_{m.hexdigest()[:10]}.cache"
    try:
        dat = np.loadtxt(fn)
        dat.shape = (-1, 2, dat.shape[-1])
        lines = [
            zb.rzline.line_from_points(*d, spline_order=1, is_sorted=True) for d in dat
        ]
        print("using cache")
        return lines
    except:
        pass
    import matplotlib.path as path
    from osa import Client

    srv2 = Client("http://esb.ipp-hgw.mpg.de:8280/services/MeshSrv?wsdl")
    # describe geometry
    mset = srv2.types.SurfaceMeshSet()  # a set of surface meshes
    mset.references = []

    # add references to single components, in this case ports
    w1 = srv2.types.SurfaceMeshWrap()
    ref = srv2.types.DataReference()
    # 371 is a vessel that includes approximations of divertor and
    # other first wall components. From a first look it doesn't seem
    # to be a high fidelity model, so maybe further improvements are
    # needed.
    # http://esb.ipp-hgw.mpg.de:8280/services/ComponentsDbRest/component/371/info
    ref.dataId = "371"  # component id
    w1.reference = ref
    # w2 = srv2.types.SurfaceMeshWrap()
    # ref = srv2.types.DataReference()
    # ref.dataId = "341" # component id
    # w2.reference = ref
    mset.meshes = [w1]  # , w2]

    lines = []
    for y in range(phi.shape[0]):
        # intersection call for phi_vals=phi
        result = srv2.service.intersectMeshPhiPlane(phi[y], mset)

        # Avoid duplicates
        all_vertices = set()

        for s, i in zip(
            result, np.arange(0, len(result))
        ):  # loop over non-empty triangle intersections
            xyz = np.array((s.vertices.x1, s.vertices.x2, s.vertices.x3)).T
            R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)[0]  # major radius
            z = xyz[:, 2][0]
            all_vertices.add((R, z))

        # Now have all vertices of vessel, but need to put them in sensible order...
        # Done by line_from_points :-)

        r, z = np.array(list(all_vertices)).T
        line = zb.rzline.line_from_points(r, z, spline_order=1, is_sorted=False)
        assert_valid(line)
        line = line.equallySpaced(n=nz)
        assert_valid(line)
        lines.append(line)

    dat = [(line.R, line.Z) for line in lines]
    dat = np.array(dat)
    dat.shape = (-1, dat.shape[-1])
    np.savetxt(fn, dat)
    return lines


## calculate curvature for curvilinear grids
def calc_curvilinear_curvature(fname, field, grid, maps):
    from scipy.signal import savgol_filter

    f = DataFile(str(fname), write=True)
    B = f.read("B")

    dx = grid.metric()["dx"]
    dz = grid.metric()["dz"]
    g_11 = grid.metric()["g_xx"]
    g_22 = grid.metric()["g_yy"]
    g_33 = grid.metric()["g_zz"]
    g_12 = 0.0
    g_13 = grid.metric()["g_xz"]
    g_23 = 0.0

    GR = np.zeros(B.shape)
    GZ = np.zeros(B.shape)
    Gphi = np.zeros(B.shape)
    dRdz = np.zeros(B.shape)
    dZdz = np.zeros(B.shape)
    dRdx = np.zeros(B.shape)
    dZdx = np.zeros(B.shape)

    for y in np.arange(0, B.shape[1]):
        pol, _ = grid.getPoloidalGrid(y)
        R = pol.R
        Z = pol.Z
        # G = \vec{B}/B, here in cylindrical coordinates
        GR[:, y, :] = field.Bxfunc(R, Z, y) / ((B[:, y, :]) ** 2)
        GZ[:, y, :] = field.Bzfunc(R, Z, y) / ((B[:, y, :]) ** 2)
        Gphi[:, y, :] = field.Byfunc(R, Z, y) / ((B[:, y, :]) ** 2)
        for x in np.arange(0, B.shape[0]):
            dRdz[x, y, :] = calc.deriv(R[x, :]) / dz[x, y, :]
            dZdz[x, y, :] = calc.deriv(Z[x, :]) / dz[x, y, :]
        for z in np.arange(0, B.shape[-1]):
            dRdx[:, y, z] = calc.deriv(R[:, z]) / dx[:, y, z]
            dZdx[:, y, z] = calc.deriv(Z[:, z]) / dx[:, y, z]

    R = f.read("R")
    Z = f.read("Z")
    dy = f.read("dy")

    ## calculate Jacobian and contravariant terms in curvilinear coordinates
    J = R * (dZdz * dRdx - dZdx * dRdz)
    Gx = (GR * dZdz - GZ * dRdz) * (R / J)
    Gz = (GZ * dRdx - GR * dZdx) * (R / J)

    G_x = Gx * g_11 + Gphi * g_12 + Gz * g_13
    G_y = Gx * g_12 + Gphi * g_22 + Gz * g_23
    G_z = Gx * g_13 + Gphi * g_23 + Gz * g_33

    dG_zdy = np.zeros(B.shape)
    dG_ydz = np.zeros(B.shape)
    dG_xdz = np.zeros(B.shape)
    dG_zdx = np.zeros(B.shape)
    dG_ydx = np.zeros(B.shape)
    dG_xdy = np.zeros(B.shape)
    for y in np.arange(0, B.shape[1]):
        for x in np.arange(0, B.shape[0]):
            dG_ydz[x, y, :] = calc.deriv(G_y[x, y, :]) / dz[x, y, :]
            dG_xdz[x, y, :] = calc.deriv(G_x[x, y, :]) / dz[x, y, :]
        for z in np.arange(0, B.shape[-1]):
            dG_ydx[:, y, z] = calc.deriv(G_y[:, y, z]) / dx[:, y, z]
            dG_zdx[:, y, z] = calc.deriv(G_z[:, y, z]) / dx[:, y, z]

    # this should really use the maps...
    for x in np.arange(0, B.shape[0]):
        for z in np.arange(0, B.shape[-1]):
            dG_zdy[x, :, z] = calc.deriv(G_z[x, :, z]) / dy[x, :, z]
            dG_xdy[x, :, z] = calc.deriv(G_x[x, :, z]) / dy[x, :, z]

    bxcvx = (dG_zdy - dG_ydz) / J
    bxcvy = (dG_xdz - dG_zdx) / J
    bxcvz = (dG_ydx - dG_xdy) / J
    bxcv = (
        g_11 * (bxcvx**2)
        + g_22 * (bxcvy**2)
        + g_33 * (bxcvz**2)
        + 2 * (bxcvz * bxcvx * g_13)
    )
    f.write("bxcvx", bxcvx)
    f.write("bxcvy", bxcvy)
    f.write("bxcvz", bxcvz)
    f.write("J", J)
    f.close()


## smooth the metric tensor components
def smooth_metric(
    fname, write_to_file=False, return_values=False, smooth_metric=True, order=7
):
    from scipy.signal import savgol_filter

    f = DataFile(str(fname), write=True)
    B = f.read("B")
    bxcvx = f.read("bxcvx")
    bxcvz = f.read("bxcvz")
    bxcvy = f.read("bxcvy")
    J = f.read("J")

    bxcvx_smooth = np.zeros(bxcvx.shape)
    bxcvy_smooth = np.zeros(bxcvy.shape)
    bxcvz_smooth = np.zeros(bxcvz.shape)
    J_smooth = np.zeros(J.shape)

    if smooth_metric:
        g13 = f.read("g13")
        g_13 = f.read("g_13")
        g11 = f.read("g11")
        g_11 = f.read("g_11")
        g33 = f.read("g33")
        g_33 = f.read("g_33")

        g13_smooth = np.zeros(g13.shape)
        g_13_smooth = np.zeros(g_13.shape)
        g11_smooth = np.zeros(g11.shape)
        g_11_smooth = np.zeros(g_11.shape)
        g33_smooth = np.zeros(g33.shape)
        g_33_smooth = np.zeros(g_33.shape)

    for y in np.arange(0, bxcvx.shape[1]):
        for x in np.arange(0, bxcvx.shape[0]):
            bxcvx_smooth[x, y, :] = savgol_filter(
                bxcvx[x, y, :], np.int(np.ceil(bxcvx.shape[-1] / 2) // 2 * 2 + 1), order
            )
            bxcvz_smooth[x, y, :] = savgol_filter(
                bxcvz[x, y, :], np.int(np.ceil(bxcvz.shape[-1] / 2) // 2 * 2 + 1), order
            )
            bxcvy_smooth[x, y, :] = savgol_filter(
                bxcvy[x, y, :], np.int(np.ceil(bxcvy.shape[-1] / 2) // 2 * 2 + 1), order
            )
            J_smooth[x, y, :] = savgol_filter(
                J[x, y, :], np.int(np.ceil(J.shape[-1] / 2) // 2 * 2 + 1), order
            )
            if smooth_metric:
                g11_smooth[x, y, :] = savgol_filter(
                    g11[x, y, :], np.int(np.ceil(g11.shape[-1] / 2) // 2 * 2 + 1), order
                )
                g_11_smooth[x, y, :] = savgol_filter(
                    g_11[x, y, :],
                    np.int(np.ceil(g_11.shape[-1] / 2) // 2 * 2 + 1),
                    order,
                )
                g13_smooth[x, y, :] = savgol_filter(
                    g13[x, y, :], np.int(np.ceil(g13.shape[-1] / 2) // 2 * 2 + 1), order
                )
                g_13_smooth[x, y, :] = savgol_filter(
                    g_13[x, y, :],
                    np.int(np.ceil(g_13.shape[-1] / 2) // 2 * 2 + 1),
                    order,
                )
                g33_smooth[x, y, :] = savgol_filter(
                    g33[x, y, :], np.int(np.ceil(g33.shape[-1] / 2) // 2 * 2 + 1), order
                )
                g_33_smooth[x, y, :] = savgol_filter(
                    g_33[x, y, :],
                    np.int(np.ceil(g_33.shape[-1] / 2) // 2 * 2 + 1),
                    order,
                )

    if write_to_file:
        # f.write('bxcvx',bxcvx_smooth)
        # f.write('bxcvy',bxcvy_smooth)
        # f.write('bxcvz',bxcvz_smooth)
        f.write("J", J_smooth)

        if smooth_metric:
            f.write("g11", g11_smooth)
            f.write("g_11", g_11_smooth)
            f.write("g13", g13_smooth)
            f.write("g_13", g_13_smooth)
            f.write("g33", g33_smooth)
            f.write("g_33", g_33_smooth)

    f.close()
    if return_values:
        return bxcvx_smooth, bxcvy_smooth, bxcvz_smooth, bxcvx, bxcvy, bxcvz


def plot_RE_poincare(
    xcentre=3, I_coil=0.005, a=0.5, start_r=3.25, start_z=0.0, npoints=100
):
    yperiod = 2 * np.pi
    field = zb.field.RotatingEllipse(
        xcentre=xcentre, I_coil=I_coil, radius=2 * a, yperiod=yperiod
    )
    zb.plot.plot_poincare(field, start_r, start_z, yperiod, revs=npoints)


def calc_iota(field, start_r, start_z):
    from scipy.signal import argrelextrema

    toroidal_angle = np.linspace(0.0, 400 * np.pi, 10000, endpoint=False)
    result = zb.fieldtracer.FieldTracer.follow_field_lines(
        field, start_r, start_z, toroidal_angle
    )
    peaks = argrelextrema(result[:, 0, 0], np.greater, order=10)[0]
    peak_locations = [result[i, 0, 0] for i in peaks]
    iota_bar = 2 * np.pi / (toroidal_angle[peaks[1]] - toroidal_angle[peaks[0]])
    return iota_bar


def plot_maps(field, grid, maps, yslice=0):
    pol, ycoord = grid.getPoloidalGrid(yslice)
    pol_next, ycoord_next = grid.getPoloidalGrid(yslice + 1)

    plt.plot(pol.R, pol.Z, "x")

    # Get the coordinates which the forward map corresponds to
    R_next, Z_next = pol_next.getCoordinate(
        maps["forward_xt_prime"][:, yslice, :], maps["forward_zt_prime"][:, yslice, :]
    )

    plt.plot(R_next, Z_next, "o")

    plt.show()


class timeit(object):
    def __init__(self, info="%f"):
        self.info = info

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        print(self.info % (time.time() - self.t0))
