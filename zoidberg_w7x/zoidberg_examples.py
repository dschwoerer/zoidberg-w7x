#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import os
import sys
import time
from functools import partial
from inspect import signature
from itertools import chain
from multiprocessing import Pool
from multiprocessing import context as mpcontext

import matplotlib.pyplot as plt
import numpy as np
import zoidberg as zb
from boututils.boutarray import BoutArray
from boututils.calculus import deriv
from boututils.datafile import DataFile
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from zoidberg.stencil_dagp_fv import doit as dagp

import fusionsc as fsc


def fsc_setup():
    from fusionsc.devices import w7x

    fsc.resolve.importOfflineData("/raven/u/dave/soft/misc1/w7x-op21.fsc")

    # Get default configuration
    config = fsc.service.LocalConfig.newMessage()

    # Adjust configuration
    config.cpuBackend.numThreads.fixed = 6

    # Apply adjusted configuration
    fsc.backends.reconfigureLocalBackend(config)


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
        iota_bar = calc_iota(field, start_r, start_z)
        if write_iota:
            f = DataFile(str(fname), write=True)
            f.write("iota_bar", iota_bar)
            f.close()
        else:
            print("Iota_bar = ", iota_bar)


def _make_w7x_polgrid(
    nx,
    ny,
    nz,
    fname,
    vmec_file,
    emc3_file,
    emc3_smooth,
    inner_VMEC,
    inner_vacuum,
    outer_VMEC,
    outer_vacuum,
    outer_vessel,
    outer_poincare,
    outer_islands,
    npoints,
    a,
    show_maps,
    show_lines,
    calc_curvature,
    smooth_curvature,
    plasma_field,
    configuration,
    vmec_url,
    field_refine,
    trace_web,
    avoid_double_outer,
    full_torus,
    min_conn_length,
    redistribute,
    return_lines,
    return_poloidal_grid,
    inner_ort,
    maxfac_inner,
    legacy_align,
    outer_fft,
    inner_fft,
    dz_relax,
    inner_poincare,
    myg,
    ycoords,
    tracer,
    symmetry,
    yperiod,
):
    try:
        with zb.zoidberg.bdata.DataFile(fname) as f:
            R = f["R"]
            Z = f["Z"]
            if R is not None and Z is not None:
                _, ny, _ = R.shape
                assert R.shape == Z.shape
                return [
                    zb.poloidal_grid.StructuredPoloidalGrid(R[:, i], Z[:, i])
                    for i in range(ny)
                ], True
    except FileNotFoundError:
        pass

    save_lines = True

    outer_lines = None
    history_outer = {}
    history_inner = {}

    if emc3_file:
        with timeit2("Aligning to EMC3 grid"):
            if isinstance(emc3_file, bool):
                emc3_file = "emc3.nc"
            nzo = 192 * 10
            if emc3_smooth is not None:
                nzo //= emc3_smooth
                print(nzo)
            outer_new = get_emc3_shape(emc3_file, "outer", ycoords, nzo)
            outer_lines = get_inner(outer_new, outer_lines)
            if show_lines or save_lines:
                history_outer["emc3"] = outer_new
            inner_lines = get_emc3_shape(emc3_file, "inner", ycoords, nz * 10)
            if show_lines or save_lines:
                history_inner["emc3"] = inner_lines

    if outer_VMEC:
        print("Aligning to outer VMEC flux surface...")
        if isinstance(outer_VMEC, bool):
            outer_VMEC = 1.5
        outer_new = get_VMEC_surfaces(
            phi=ycoords, s=outer_VMEC, npoints=nz * 10, w7x_run=vmec_url
        )
        outer_lines = get_inner(outer_new, outer_lines)
        if show_lines or save_lines:
            history_outer["vmec"] = outer_new

    if outer_vessel:
        with timeit2("Aligning to plasma vessel ..."):
            outer_new = get_W7X_vessel(phi=ycoords, nz=nz * 10)
            outer_lines = get_inner(outer_new, outer_lines)
            assert_valid(outer_lines)
        if show_lines or save_lines:
            history_outer["vessel"] = outer_new

    if outer_poincare:
        R0 = {0: 6.3, 4: 6.17, 20: 6.214, 21: 6.2246}[configuration]
        with timeit2(f"Aligning to poincare R={R0}"):
            outer_new = get_W7X_poincare(
                R0, phi=ycoords, conf=configuration, nz=nz * 10, symmetry=symmetry
            )
            outer_lines = get_inner(outer_new, outer_lines)
        if show_lines or save_lines:
            history_outer["poincare"] = outer_new

    if inner_poincare:
        R0 = {0: 6.12335, 3: 6.16, 20: 6.12335, 21: 6.133}[configuration]
        with timeit2(f"Aligning to poincare R={R0}"):
            inner_lines = get_W7X_poincare(
                R0, phi=ycoords, conf=configuration, nz=nz * 10, symmetry=symmetry
            )
            if show_lines or save_lines:
                history_inner["poincare"] = inner_lines

    if outer_islands is not False:
        import shapely
        import shapely.geometry as sg

        if outer_islands is True:
            outer_islands = 0.1
        R0, Z0 = {0: (5.698742375633017 - 1e-4, 0), 3: (6.2435, 0.05)}[configuration]
        with timeit2("tracing islands for poincare"):
            islands = trace_poincare([R0], [Z0], ycoords, 200, configuration, symmetry)
        with timeit2("calculating islands shape"):
            outer_new = []
            for yid, island in enumerate(islands.transpose("phi", ...)):
                island = island.transpose(..., "Rz")
                island_data = island.values
                island_data.shape = (-1, 2)
                extend = outer_islands
                if extend < 0.1:
                    extend = 0.1
                points = [sg.Point(*i).buffer(extend) for i in island_data]
                xy = shapely.union_all(points)
                if outer_islands < extend:
                    xy = xy.buffer(outer_islands - extend)
                xy = xy.exterior.xy
                cur = zb.rzline.RZline(*xy, smooth=50)
                cur = cur.equallySpaced(n=nz * 10)
                outer_new.append(cur)
            outer_lines = get_inner(outer_new, outer_lines)
            assert_valid(outer_lines)
            if show_lines or save_lines:
                history_outer["island"] = outer_new

    if outer_vacuum or inner_vacuum:
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
            if show_lines or save_lines:
                history_inner["vacuum"] = inner_lines
        if outer_vacuum:
            print("Aligning to outer vacuum flux surface...")
            outer_lines = get_lines(
                field, start_r + a, start_z, ycoords, yperiod=yperiod, npoints=npoints
            )
            if show_lines or save_lines:
                history_outer["vacuum"] = outer_lines

    xmin = np.min([min(outer_lines[i].R) for i in range(ny)])
    xmax = np.max([max(outer_lines[i].R) for i in range(ny)])
    zmin = np.min([min(outer_lines[i].Z) for i in range(ny)])
    zmax = np.max([max(outer_lines[i].Z) for i in range(ny)])

    if inner_VMEC:
        with timeit2("Aligning to inner VMEC flux surface"):
            if inner_VMEC is True:
                inner_VMEC = 0.67
            inner_lines = get_VMEC_surfaces(
                phi=ycoords, s=inner_VMEC, npoints=nz * 2, w7x_run=vmec_url
            )
            if show_lines or save_lines:
                history_inner["vmec"] = inner_lines

    check_VMEC = True
    if inner_VMEC and check_VMEC:
        import shapely.geometry as sg

        assert np.isclose(inner_lines[0].Z[0], 0)
        tracings = get_W7X_poincare(
            [inner_lines[0].R[0]],
            phi=ycoords,
            conf=configuration,
            nz=nz,
            symmetry=symmetry,
        )
        history_inner["poincare"] = tracings
        errors = []
        for yid, vmectracing in enumerate(zip(inner_lines, tracings)):
            vt = vmectracing
            vmectracing = [sg.Polygon(np.array([l1.R, l1.Z]).T) for l1 in vmectracing]
            vmec, tracing = vmectracing
            assert vmec.is_valid
            if not tracing.is_valid:
                from zoidberg.spectralline import SpectralLine

                l1 = vt[1]
                print([type(x) for x in vt])
                l1 = SpectralLine(l1, 16).equallySpaced(len(l1.R))
                tracing = sg.Polygon(np.array((l1.R, l1.Z)).T)
                print(tracing.is_valid)
                # tracing.plot()
            error = vmec.symmetric_difference(tracing).area / vmec.area
            errors.append(error)
            if error > 2e-2:
                for d, lbl in zip(vt, ("vmec", "tracing")):
                    plt.plot(d.R, d.Z, "-", label=lbl)
                    plt.plot(d.R, -d.Z, "-", label=lbl + " ud")
                plt.plot(vt[1].R, vt[1].Z, "x")
                plt.title(f"config={configuration} ; y_id={yid} ; error={error}")
                plt.legend()
                plt.show()
                raise ValueError(f"Error is {error} and larger than expected.")
        print(f"vmec vs fieldline error is {np.max(errors)*100:.4f} %")

    if outer_fft:
        from zoidberg.spectralline import SpectralLine

        onz = len(outer_lines[0].R)
        outer_new = [SpectralLine(x, outer_fft).equallySpaced(onz) for x in outer_lines]
        if show_lines or save_lines:
            history_outer["fft"] = outer_new
        outer_lines = outer_new

    if inner_fft:
        from zoidberg.spectralline import SpectralLine

        onz = len(inner_lines[0].R)
        inner_new = [SpectralLine(x, inner_fft).equallySpaced(onz) for x in inner_lines]
        if show_lines or save_lines:
            history_inner["fft"] = inner_new
        inner_lines = inner_new

    if avoid_double_outer:
        print("Avoid double boundary")

        def trace(outer_lines, num=1):
            m = hashlib.md5()
            m.update("-".join([str(x) for x in ycoords]).encode())
            fnn = f"w7x_vessel_{nz}{f'_{num}' if num > 1 else ''}_{myhash(ycoords, *outer_lines)}.cache"
            fn = f"w7x_vessel_{nz}_{m.hexdigest()[:10]}.cache" if num == 1 else None
            try:
                dat = np.loadtxt(fnn)
                dat.shape = (2, num, len(ycoords), 2, dat.shape[-1])
                return dat
            except FileNotFoundError:
                try:
                    dat = np.loadtxt(fn)
                    np.savetxt(fnn, dat)
                    dat.shape = (2, num, len(ycoords), 2, dat.shape[-1])
                    return dat
                except:  # noqa
                    print("nocache:", fnn)

            dy = ycoords[1] - ycoords[0]
            dat = [[[] for _ in range(num)] for _ in range(2)]
            with Pool(6) as pool:
                tracerfunc = partial(
                    tracer.follow_field_lines, chunk=int(1e8), timeout=60, retry=2
                )
                for outer, y in zip(outer_lines, ycoords):
                    for n in range(num):
                        for k, dist in enumerate([dy, -dy]):
                            dat[k][n] += [
                                pool.apply_async(
                                    tracerfunc,
                                    (outer.R, outer.Z, [y, y + dist * (n + 1)]),
                                )
                            ]
                with tqdm(
                    total=len(dat) * len(dat[0][0]) * len(dat[0]), desc="Tracing4DB"
                ) as prog:
                    for dai in dat:
                        for da in dai:
                            for i in range(len(da)):
                                da[i] = np.transpose(da[i].get()[1])
                                prog.update()

            dat = np.array(dat)
            shape = dat.shape
            dat.shape = (-1, shape[-1])
            np.savetxt(fnn, dat)
            dat.shape = shape
            return dat

        fwds, bwds = trace(outer_lines, num=min_conn_length)
        for i in range(len(fwds)):
            fwds[i] = np.roll(fwds[i], i + 1, axis=0)
            bwds[i] = np.roll(bwds[i], -i - 1, axis=0)
        assert_valid(fwds)
        assert_valid(bwds)

        def toline(rzs):
            return [
                zb.rzline.line_from_points(
                    *np.array(rz), spline_order=1, is_sorted=True
                )
                for rz in rzs
            ]

        for i in range(min_conn_length):
            history_outer[f"avoid_short_fwd_{i+1}"] = toline(fwds[i])
            history_outer[f"avoid_short_bwd_{i+1}"] = toline(bwds[i])
            for j in range(min_conn_length):
                if i + j < min_conn_length:
                    union = get_outer(fwds[i], bwds[j])
                    history_outer[f"avoid_short_{i+1}_-{j+1}"] = union
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
    if save_lines:
        tosave = dict(
            inner=inner_lines,
            outer=outer_lines,
        )
        import xarray as xr

        ds = xr.Dataset()
        for k, v in history_outer.items():
            tosave[f"outer_{k}"] = v
        for k, v in history_inner.items():
            tosave[f"inner_{k}"] = v
        for theta, post in (
            (np.linspace(0, np.pi * 2, nz * 10, endpoint=False), "_highres"),
            (np.linspace(0, np.pi * 2, nz, endpoint=False), ""),
        ):
            print(len(theta), theta[-1], post)
            for k, v in tosave.items():
                vn = np.empty((2, ny, len(theta)))
                for i in range(ny):
                    vn[:, i] = v[i].position(theta)
                ds[k + post] = ("RZ", "ny", "nz" + post), vn
        linesname = fname[:-3] + ".lines.nc"
        ds.to_netcdf(linesname)
        print(f"lines info written to {linesname}")

    if return_lines:
        return inner_lines, outer_lines
    with timeit2("Creating poloidal grids"):
        poloidal_grid = [
            zb.poloidal_grid.grid_elliptic(
                inner,
                outer,
                nx,
                nz,
                show=show_maps,
                nx_outer=2,
                nx_inner=2,
                inner_ort=inner_ort,
                legacy_align=legacy_align,
                maxfac_inner=maxfac_inner,
                dz_relax=dz_relax,
            )
            for inner, outer in zip(inner_lines, outer_lines)
        ]

    if return_poloidal_grid:
        return poloidal_grid

    if redistribute:

        def getWeights(Vi, n):
            med = np.median(Vi)
            Vi[Vi < med] = med
            t1 = np.linspace(0, 1, len(Vi) + 1)
            return interp1d(t1, np.append(Vi, Vi[0]), assume_sorted=True)(
                np.linspace(0, 1, n, endpoint=False)
            )

        def doredistribute(loi, pmi, ind=-3):
            RZ1 = np.array((loi.R, loi.Z))
            RZ2 = np.array((pmi.R, pmi.Z))[:, ind]
            drz = (RZ1[..., None] - RZ2[:, None, :]) ** 2
            drz = np.sum(drz, axis=0)
            lr1 = RZ1.shape[1]
            lr2 = RZ2.shape[1]
            fac = lr1 / lr2
            j = np.arange(lr2, dtype=int)
            sums = [
                np.sum(drz[np.round(j * fac).astype(int) - i, j]) for i in range(lr1)
            ]
            nind = -np.argmin(sums)
            if sums[-nind] > 10:
                print("plotting")
                plt.figure()
                plt.plot(sums)
                plt.plot([-nind, -nind], plt.ylim())
                plt.show()
                assert (
                    False
                ), "Expected to find a minimum close to zero, but failed to do so"
            m = pmi.metric()
            V = m["J"] * m["dx"] * m["dz"]
            weights = getWeights(V[ind], len(loi.R))
            lne = loi.equallySpaced(weights=np.roll(weights ** (redistribute), nind))
            return lne

        with timeit2(f"Minimising grid cells variation with exponent {redistribute}"):
            inner_lines = [
                doredistribute(*a, 0) for a in zip(inner_lines, poloidal_grid)
            ]
            outer_lines = [
                doredistribute(*a, -3) for a in zip(outer_lines, poloidal_grid)
            ]

        with timeit2("Creating poloidal grids"):
            poloidal_grid = [
                zb.poloidal_grid.grid_elliptic(
                    inner, outer, nx, nz, show=show_maps, nx_outer=2
                )
                for inner, outer in zip(inner_lines, outer_lines)
            ]

    return poloidal_grid, False


def W7X(
    nx=68,
    ny=32,
    nz=256,
    fname="W7-X.fci.nc",
    vmec_file="w7-x.wout.nc",
    emc3_file=None,
    emc3_smooth=None,
    inner_VMEC=False,
    inner_vacuum=False,
    outer_VMEC=False,
    outer_vacuum=False,
    outer_vessel=False,
    outer_poincare=False,
    outer_islands=False,
    npoints=100,
    a=2.5,
    show_maps=True,
    show_lines=False,
    calc_curvature=False,
    smooth_curvature=False,
    plasma_field=False,
    configuration=0,
    vmec_url=None,
    field_refine=1,
    trace_web=True,
    avoid_double_outer=True,
    full_torus=False,
    min_conn_length=3,
    redistribute=0,
    return_lines=False,
    return_poloidal_grid=False,
    inner_ort=True,
    maxfac_inner=3,
    legacy_align=False,
    outer_fft=None,
    inner_fft=None,
    dz_relax=None,
    inner_poincare=False,
    myg=1,
):
    try:
        kwargs = {k: locals()[k] for k in signature(W7X).parameters.keys()}
    except:  # noqa
        print("Failed to record arguments")
        kwargs = {}

    if vmec_url is None:
        urls = {
            0: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0000_+0000/01/00jh_l/wout.nc",
            3: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0750_+0750/01/00/wout.nc",
            4: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_-0690_-0690/01/00/wout.nc",
            5: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1042_1042_1127_1127_+0000_+0000/01/00/wout.nc",
            6: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/0972_0926_0880_0852_+0000_+0000/01/00jh/wout.nc",
            20: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_-0246_-0246/01/000/wout.nc",
            21: "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_-0264_-0264/01/000/wout.nc",
            "FMM002": "http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_-0264_-0264/01/000/wout.nc",
        }
        if configuration not in urls:
            raise KeyError(
                f"Do not know the appropriate vmec url for configuartion '{configuration}'.\n"
                "Check http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/"
            )
        vmec_url = urls[configuration]

    symmetry = 1 if full_torus else 5
    yperiod = 2 * np.pi / symmetry
    ycoords = np.linspace(0.0, yperiod, ny, endpoint=False)

    avoidfsc = True  # False
    with timeit2("Creating a field"):
        if field_refine:
            raise RuntimeError("Not implemented anymore")
        else:
            if avoidfsc:
                field = zb.field.W7X_vacuum_on_demand(configuration)
            else:
                fsc_setup()
                assert configuration == 0
                fsc_field = (
                    fsc.devices.w7x.standard()
                )  # + w7x.controlCoils([2000, 2000])
                fsc_geometry = fsc.devices.w7x.op21Geometry()

                fsc_grid = fsc.devices.w7x.defaultGrid()
                fsc_grid.nR = 128
                fsc_grid.nZ = 128
                fsc_grid.nPhi = 32

                fsc_field = fsc_field.compute(fsc_grid)

                field = zb.field.FusionSC(fsc_field)

    with timeit2("Getting tracer took"):
        if avoidfsc:
            tracer = (
                zb.fieldtracer.FieldTracerWeb(
                    configId=configuration, stepsize=1e-2, retry=100
                )
                if trace_web
                else None
            )
            tracer = zb.fieldtracer.CachingFieldTracer(tracer)
        else:
            # fsc_mapping = fsc.flt.computeMapping(
            #     fsc_field,
            #     np.radians([72]),  # Mapping planes
            #     r=np.linspace(4, 7, 200),  # R values for mapping grid
            #     z=np.linspace(-1.5, 1.5, 200),  # Z values for mapping grid
            #     stepSize=0.01,  # Step size for tracing computation
            #     distanceLimit=7
            #     * 2
            #     * np.pi
            #     / 8,  # Distance limit for tracing computation (no field line needed to be longer than 1/10th of device)
            #     toroidalSymmetry=5,
            # )

            tracer = zb.fieldtracer.FusionSC(
                fsc_field,
                grid=fsc_grid,
                # geometry=fsc_geometry,
                # geometryGrid=fsc.devices.w7x.defaultGeometryGrid(),
                # mapping=fsc_mapping,
                stepSize=0.01,
                distanceLimit=1,
                # targetError=1e-3,
                # maxStepSize=0.2,
            )

    ret = _make_w7x_polgrid(
        nx,
        ny,
        nz,
        fname,
        vmec_file,
        emc3_file,
        emc3_smooth,
        inner_VMEC,
        inner_vacuum,
        outer_VMEC,
        outer_vacuum,
        outer_vessel,
        outer_poincare,
        outer_islands,
        npoints,
        a,
        show_maps,
        show_lines,
        calc_curvature,
        smooth_curvature,
        plasma_field,
        configuration,
        vmec_url,
        field_refine,
        trace_web,
        avoid_double_outer,
        full_torus,
        min_conn_length,
        redistribute,
        return_lines,
        return_poloidal_grid,
        inner_ort,
        maxfac_inner,
        legacy_align,
        outer_fft,
        inner_fft,
        dz_relax,
        inner_poincare,
        myg,
        ycoords,
        tracer,
        symmetry,
        yperiod,
    )
    if return_lines or return_poloidal_grid:
        return ret
    poloidal_grid, loaded = ret

    # Create the 3D grid by putting together 2D poloidal grids
    with timeit2("Creating a grid"):
        grid = zb.grid.Grid(poloidal_grid, ycoords, yperiod, yperiodic=True)

    with zb.zoidberg.MapWriter(str(fname), create=(not loaded)) as mw:
        if not loaded and "g33":
            with timeit2("Calculating metric terms"):
                grid.metric()
        mw.add_grid_field(grid, field)

        with timeit2("DAGP"):
            if not (loaded and "dagp_fv_volume" in mw.f.list()):
                mw.write_dict(dagp(poloidal_grid))

        mw.write_dict(
            {
                "phi": BoutArray(ycoords, dict(bout_type="ArrayY")),
                "y": BoutArray(ycoords, dict(bout_type="ArrayY")),
                "w7x_configuration_id": configuration,
            }
        )

        with timeit2("Creating and Writing maps"):
            haskeys = loaded
            if haskeys:
                for offset in chain(range(1, myg + 1), range(-1, -(myg + 1), -1)):
                    for k in ["R", "Z", "xt_prime", "zt_prime"]:
                        if (
                            zb.zoidberg.parallel_slice_field_name(k, offset)
                            not in mw.f.list()
                        ):
                            haskeys = False
            if haskeys:
                maps = {}
                for offset in chain(range(1, myg + 1), range(-1, -(myg + 1), -1)):
                    for k in ["R", "Z"]:
                        key = zb.zoidberg.parallel_slice_field_name(k, offset)
                        maps[key] = mw.f[key]
                for k in ["R", "Z"]:
                    maps[k] = mw.f[k]
                mw._write_par_metric(maps, myg, ycoords)
                print(ycoords)
                mw._final()
            else:
                if show_maps:
                    maps = zb.make_maps(grid, field, field_tracer=tracer, nslice=myg)
                    mw.add_maps(maps)
                else:
                    mw.add_maps(
                        zb.make_maps(grid, field, field_tracer=tracer, nslice=myg)
                    )

        ops = {}
        for k, v in kwargs.items():
            if isinstance(v, bool):
                v = int(v)
            if v is None:
                v = str(None)

            ops[f"option:{k}"] = v
        mw.write_dict(ops)

    with timeit("Calculating curvature took %f"):
        if calc_curvature:
            print("calculating curvature...")
            calc_curvilinear_curvature(fname, field, grid, maps)

    with timeit("Smoothing curvature took %f"):
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
        for ln in line:
            assert_valid(ln)
        return
    if isinstance(line, np.ndarray):
        if len(line.shape) > 2:
            for ln in line:
                assert_valid(ln)
            return
        if line.shape[0] == 2:
            x, y = line
        else:
            x, y = line.T
    else:
        x = line.R
        y = line.Z
    import shapely.geometry as sg

    pi = sg.Polygon(np.array((x, y)).T)
    if pi.is_valid:
        return
    dx = x - np.roll(x, 1)
    dy = y - np.roll(y, 1)
    dr = dx**2 + dy**2
    bad = dr == 0
    print(np.arange(len(bad))[bad])
    plt.plot(np.roll(x - 5, 10), label="R - 5")
    plt.plot(np.roll(y, 10), label="z")
    plt.plot(dx * 1000, label="dR")
    plt.plot(dy * 1000, label="dz")
    plt.plot((dx**2 + dy**2) * 1e6, label="ds²")
    plt.legend()

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
        l10, l20 = l1, l2
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
            except:  # noqa
                li0 = l10 if np.all(li == l1) else l20
                plt.plot(*li0.T, "x-", label="input")
                plt.plot(*li.T, "x-", label="used")
                plt.legend()
                plt.title("line")
                plt.figure()
                plt.plot(li.T[0], "x-")
                plt.plot(li.T[1], "o-")
                plt.title("R and z components")
                x, y = li.T
                dx = x - np.roll(x, 1)
                dy = y - np.roll(y, 1)
                plt.figure()
                plt.scatter(dx, dy)
                plt.title("dR and dz components")
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
                best = None
                print(
                    f"io={io}; Got multiple Polygons with areas",
                    [x.area for x in res.geoms],
                )
                for r in res.geoms:
                    plt.plot(*np.array(r.exterior.coords).T, label=r.area)
                    if best is None or r.area > best.area:
                        best = r
                plt.legend()
                plt.show()
                res = best
            newrz = res.exterior.coords
        except:  # noqa
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
        _cache["tracer"] = Client(
            "http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl"
        )
    return _cache["tracer"]


def connection_length(RZ, phi, conf, limit):
    """
    Calculate the connection length.

    RZ: RZ coorindates

    """
    import xarray as xr

    fn = f"w7x_poincare_{conf}_{myhash(*RZ, phi)}.cache.nc"
    try:
        return xr.open_dataset(fn)
    except FileNotFoundError:
        print("nocache:", fn)

    R, Z = RZ
    dats = [np.ravel(R * np.cos(phi)), np.ravel(R * np.sin(phi)), np.ravel(Z)]

    chunk = max(12, int(30 / limit))

    with Pool(6) as pool:

        def start(i):
            return pool.apply_async(
                _connection_length,
                (
                    [d[i : i + chunk] for d in dats],
                    conf,
                    limit,
                ),
            )

        # Start parallel calculation
        results = [start(i) for i in range(0, len(dats[0]), chunk)]
        # Wait for result and combine
        done = np.zeros(len(results), dtype=bool)
        with tqdm(total=len(done)) as pbar:
            j = 0
            while not np.all(done):
                if done[j]:
                    continue
                try:
                    results[j] = results[j].get(timeout=5)
                    done[j] = True
                    pbar.update(1)
                except mpcontext.TimeoutError:
                    pass
                j += 1
                if j == len(done):
                    j = 0
                elif np.sum(1 - done[:j]) > os.cpu_count():
                    j = 0

            try:
                results = np.concatenate(results, axis=2)
            except ValueError:
                print(results)
                raise

    out = results

    out.shape = (4, 2, *R.shape)
    ds = xr.Dataset()
    dims = ("tracing_direction",)
    try:
        dims = *dims, *R.dims
    except AttributeError:
        dims = *dims, *[f"dim_{d}" for d in range(len(R.shape))]
    ds["con_length"] = list(dims), out[0]
    ds["end_point"] = list(("spatial", *dims)), out[1:]

    try:
        coords = R.coords
    except AttributeError:
        coords = {}
    ds = ds.assign_coords(
        tracing_direction=["forward", "backwards"], spatial=list("xyz"), **coords
    )
    ds.to_netcdf(fn)
    return ds


def _connection_length(xyz, conf, limit, dir=None):
    from time import sleep

    sleep(np.random.random(1)[0] * 10)

    flt = get_tracer()
    pnts = flt.types.Points3D()
    x, y, z = xyz
    pnts.x1 = x
    pnts.x2 = y
    pnts.x3 = z

    config = flt.types.MagneticConfig()
    config.configIds = [conf]

    if 0:
        g = flt.types.Grid()
        my_grid = flt.types.CylindricalGrid()

        my_grid.RMin = 4.05
        my_grid.RMax = 6.75
        my_grid.ZMin = -1.35
        my_grid.ZMax = 1.35
        my_grid.numR = 181
        my_grid.numZ = 181
        my_grid.numPhi = 96
        g.cylindrical = my_grid
        g.fieldSymmetry = 5
        config.grid = g

    task = flt.types.Task()
    task.step = 1e-2
    con = flt.types.ConnectionLength()
    con.limit = limit
    con.returnLoads = False
    task.connection = con

    machine = flt.types.Machine(1)
    machine.grid.numX, machine.grid.numY, machine.grid.numZ = 500, 500, 100
    machine.grid.ZMin, machine.grid.ZMax = -1.5, 1.5
    machine.grid.YMin, machine.grid.YMax = -7, 7
    machine.grid.XMin, machine.grid.XMax = -7, 7

    machine.assemblyIds = []  # 371]  # [2, 5, 6, 7, 9, 15, 16, 17]  # [1]
    machine.meshedModelsIds = [371]
    task.machine = machine

    timeout = x.size * limit + 10
    if isinstance(dir, bool):
        dir = (dir,)
    dir = dir or (False, True)
    out = np.empty((4, len(dir), x.size)) * np.nan
    for config.inverseField in dir:
        res = flt.service.trace(
            pnts, config, task, machine, osa_retry=5, osa_timeout=timeout
        )
        for i, con in enumerate(res.connection):
            out[0, dir.index(config.inverseField), i] = con.length
            if con.element is not None:
                for j, var in enumerate((con.x, con.y, con.z)):
                    out[j + 1, dir.index(config.inverseField), i] = var

    return out


def trace_boundary(fn, conf):
    with DataFile(fn, write=True) as dat:
        nx = dat["nx"]
        mxg = dat.get("MXG", 2)
        myg = dat.get("MYG", 1)
        phi = dat["phi"][None, :, None]
        limit = np.max(dat["R"]) * np.max(dat["dy"]) * (myg + 1.0)

        xyz = dat["R"] * np.cos(phi), dat["R"] * np.sin(phi), dat["Z"]
        outs = np.empty((4, *dat["R"].shape))
        for fwd, dir in zip(("forward", "backward"), (False, True)):
            xtp = dat[fwd + "_xt_prime"]
            bnd = xtp > nx - mxg - 1
            bnd[-mxg:] = False
            if 0:
                while np.sum(bnd) > 100:
                    bnd = np.logical_and(bnd, np.random.random(xyz[0].shape) > 0.5)
                    print(np.sum(bnd))
            print(np.sum(bnd))
            print(limit, type(limit))
            out = _connection_length([x[bnd] for x in xyz], conf, float(limit), dir)
            print("bad", np.sum(~np.isfinite(out[1])))

            outs *= np.nan
            outs[:, bnd] = out[:, 0, :]
            dat.write(f"{fwd}_length", outs[0] / dat["R"] / dat["dy"])
            for d, i in zip("xyz", range(1, 5)):
                dat.write(f"{fwd}_end_{d}", outs[i] / dat["R"] / dat["dy"])
            print(out)

        return


def trace_poincare(start_r, start_z, yslices, npoints, conf, symmetry=5, step=1e-2):
    import xarray as xr

    start_r = np.array(start_r)
    start_z = np.array(start_z)

    assert yslices[0] == 0
    fn = f"w7x_poincare_{conf}_{myhash(start_r, start_z, yslices, [symmetry, npoints, step])}.cache.nc"
    try:
        with xr.open_dataarray(fn) as dat:
            return dat
    except:  # noqa
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

    config = flt.types.MagneticConfig()
    config = zb.field._set_config(config, conf)

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
                    assert np.isclose(
                        yslices0[i1], s.phi0 % (2 * np.pi / symmetry)
                    ), f"{yslices0[i1]} is not close to {s.phi0 % (2 * np.pi / symmetry)}"
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


def get_W7X_poincare(R, phi, conf, nz, symmetry=5):
    dat = _get_W7X_poincare(R, phi, conf, symmetry)
    out = []
    for cur in dat.transpose("phi", ...):
        cur = zb.rzline.RZline(*cur)
        cur = cur.equallySpaced(n=nz)
        out.append(cur)
    return out


def _get_W7X_poincare(R, phi, conf, symmetry):
    import xarray as xr

    fn = f"w7x_poincare2_{conf}_{myhash(R, phi, conf)}.cache.nc"
    try:
        dat = xr.open_dataarray(fn)
        return dat
    except:  # noqa
        pass

    pnc = trace_poincare([R], [0], phi, 50, conf, symmetry=symmetry)

    def _prod(a):
        r = 1
        for k in a:
            r *= k
        return r

    newshape = (pnc.shape[0], pnc.shape[1], np.prod(pnc.shape[2:]))
    out = np.empty(newshape)
    for i, phi in enumerate(pnc.phi):
        rz = pnc.sel(phi=phi)
        cur = zb.rzline.line_from_points(*[k.data.flatten() for k in rz], smooth=50)
        out[:, i, :] = np.array([cur.R, cur.Z])

    da = xr.DataArray(
        out, dims=(*pnc.dims[:2], "points"), coords=dict(Rz=pnc["Rz"], phi=pnc["phi"])
    )
    da.to_netcdf(fn)
    return da


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
    fn = f"w7x_vessel_{nz}_{myhash(phi)}.cache"
    try:
        dat = np.loadtxt(fn)
        dat.shape = (-1, 2, dat.shape[-1])
        lines = [
            zb.rzline.line_from_points(*d, spline_order=1, is_sorted=True) for d in dat
        ]
        return lines
    except:  # noqa
        pass
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
        last = None
        rzs = []
        for rz in zip(line.R, line.Z):
            if last is not None:
                ds = np.array(last) - np.array(rz)
                ds2 = np.sum(ds**2)
                if ds2 < 1e-9:
                    continue
            last = rz
            rzs.append(rz)
        line = zb.rzline.line_from_points(
            *np.array(rzs).T, spline_order=1, is_sorted=True
        )
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
    f = DataFile(str(fname), write=True)
    B = f.read("B")

    metric = grid.metric()
    dx = metric["dx"]
    dz = metric["dz"]
    g_11 = metric["g_xx"]
    g_22 = metric["g_yy"]
    g_33 = metric["g_zz"]
    g_12 = 0.0
    g_13 = metric["g_xz"]
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
            dRdz[x, y, :] = deriv(R[x, :]) / dz[x, y, :]
            dZdz[x, y, :] = deriv(Z[x, :]) / dz[x, y, :]
        for z in np.arange(0, B.shape[-1]):
            dRdx[:, y, z] = deriv(R[:, z]) / dx[:, y, z]
            dZdx[:, y, z] = deriv(Z[:, z]) / dx[:, y, z]

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
            dG_ydz[x, y, :] = deriv(G_y[x, y, :]) / dz[x, y, :]
            dG_xdz[x, y, :] = deriv(G_x[x, y, :]) / dz[x, y, :]
        for z in np.arange(0, B.shape[-1]):
            dG_ydx[:, y, z] = deriv(G_y[:, y, z]) / dx[:, y, z]
            dG_zdx[:, y, z] = deriv(G_z[:, y, z]) / dx[:, y, z]

    # this should really use the maps...
    for x in np.arange(0, B.shape[0]):
        for z in np.arange(0, B.shape[-1]):
            dG_zdy[x, :, z] = deriv(G_z[x, :, z]) / dy[x, :, z]
            dG_xdy[x, :, z] = deriv(G_x[x, :, z]) / dy[x, :, z]

    bxcvx = (dG_zdy - dG_ydz) / J
    bxcvy = (dG_xdz - dG_zdx) / J
    bxcvz = (dG_ydx - dG_xdy) / J

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


def myhash(*args, len=10):
    """
    Compute a digest over some numerical data
    """
    m = hashlib.md5()
    for arg in args:
        if isinstance(arg, zb.rzline.RZline):
            m.update("-".join([str(x) for x in zip(arg.R, arg.Z)]).encode())
        else:
            m.update("-".join([str(x) for x in np.array([arg]).ravel()]).encode())
    return m.hexdigest()[:len]


def calc_iota(field, start_r, start_z):
    from scipy.signal import argrelextrema

    toroidal_angle = np.linspace(0.0, 400 * np.pi, 10000, endpoint=False)
    result = zb.fieldtracer.FieldTracer.follow_field_lines(
        field, start_r, start_z, toroidal_angle
    )
    peaks = argrelextrema(result[:, 0, 0], np.greater, order=10)[0]
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


def get_emc3_shape_simple(ds, direction, phi, coords, zsign):
    import xarray as xr

    # if "zone" in ds.dims:
    def getne(di):
        return di.emc3.sel(phi=phi).ne

    rdims = ds["_r_dims"] if "_r_dims" in ds else None
    nes = [getne(di) for di in ds.emc3.iter_zones()]
    if rdims is not None:
        for i, ri in enumerate(rdims):
            if len(nes[i].r) > ri:
                nes[i] = nes[i].isel(r=slice(None, ri))

    # For some reason checking for finite values fails sometimes
    ridss = [np.argmax(x.values[::direction] > 0, axis=0) for x in nes]
    if direction == -1:
        ridss = [len(ne.r) - x for x, ne in zip(ridss, nes)]
    # for theta
    coords_ = [c.emc3.sel(phi=phi) for c in coords.emc3.iter_zones()]
    coords_ = [xr.Dataset({k: c.emc3[f"{k}_corners"] for k in "Rz"}) for c in coords_]

    for c in coords_:
        c.z.values *= zsign

    def myrange(a, b):
        if a > b:
            return range(a, b - 1, -1)
        return range(a, b + 1)

    rzs = []
    for rids, coords__ in zip(ridss, coords_):
        out = []
        last = rids[-1]
        for tid, rid in enumerate(rids):
            for cr in myrange(last, rid):
                out.append((tid, cr))
            last = rid
        out = np.array(out).T
        out_ = [xr.DataArray(o, dims="index") for o in out]
        c__ = coords__.isel(theta=out_[0], r=out_[1])
        rzs.append(np.array([c__[k].values for k in "Rz"]))
    return rzs


def get_emc3_shape(fn, direction, ycoords, nz, return_early=None, period=5):
    import xarray as xr
    import zoidberg as zb

    fnc = f"w7x_{fn.replace('/','_')}_{direction}_{myhash(ycoords, nz)}.cache.nc"
    try:
        with xr.open_dataarray(fnc) as da:
            pass
        return [zb.rzline.RZline(*cur, spline_order=True) for cur in da]

    except FileNotFoundError:
        pass

    import xemc3  # noqa
    from zoidberg.smooth import smooth

    try:
        from scipy.interpolate import make_smoothing_spline
    except ImportError:
        raise ImportError("scipy is too old!")

    direction = dict(outer=-1, inner=+1)[direction]

    with xr.open_dataset(fn) as ds:
        pass

    coords = xr.Dataset({k: ds[k] for k in [f"{k}_bounds" for k in "Rz"]})
    for d in ds:
        if d.startswith("_") and d.endswith("_dims"):
            coords[d] = ds[d]

    lines = []

    keys = ["ne", "R_bounds", "z_bounds", "phi_bounds"]
    for k in ds:
        if k.startswith("_") and k.endswith("_dims"):
            keys.append(k)
    ds = xr.Dataset({k: ds[k] for k in keys})
    print("ds done")
    points = []
    phimax = np.nanmax(ds.phi_bounds)
    assert (
        np.isclose(phimax, np.pi / period) or np.isclose(phimax, np.pi * 2 / period),
    ) and np.isclose(
        np.nanmin(ds.phi_bounds), 0
    ), f"Expected 0 <= ds.phi_bounds <= {np.pi/period} but got {np.nanmin(ds.phi_bounds)} ... {np.nanmax(ds.phi_bounds)}"
    for phi in ycoords:
        phi %= np.pi * 2 / period
        if phi <= phimax:
            zsign = +1
        else:
            phi = 2 * np.pi / period - phi
            zsign = -1

        if "zone" in ds.dims:
            import shapely as sg

            rzsi = get_emc3_shape_simple(ds, +1, phi, coords, zsign)
            rzso = get_emc3_shape_simple(ds, -1, phi, coords, zsign)
            pgs = [sg.Polygon(o.T, [i.T]) for o, i in zip(rzso, rzsi)]

            import shapely.plotting

            assert all([p.is_valid for p in pgs])
            try:
                pg = sg.union_all(pgs, grid_size=1e-8)
            except sg.lib.GEOSException:
                pgs = [pg.buffer(1e-8) for pg in pgs]
                try:
                    pg = sg.union_all(pgs, grid_size=1e-8)
                except sg.lib.GEOSException:
                    pgs = [pg.buffer(1e-6) for pg in pgs]
                    pg = sg.union_all(pgs, grid_size=1e-6)

            if direction == -1:
                if isinstance(pg, sg.MultiPolygon):
                    pgs = sorted([(x.area, x) for x in enumerate(pg.geoms)])
                    pg = pgs[-1][1][1]
                rzs = pg.exterior
            else:
                if isinstance(pg, sg.MultiPolygon):
                    pgs = pg.geoms
                    inner = []
                    for pg in pgs:
                        inner += pg.interiors
                else:
                    inner = pg.interiors
                inner = [sg.Polygon(x) for x in inner]
                if len(inner) > 1:
                    areas = [x.area for x in inner]
                    inner = sorted(zip(areas, enumerate(inner)))
                    print(f"Warning! Need to pick from {len(inner)} surfaces!", inner)
                    rzs = inner[-1][1][1]
                else:
                    rzs = inner[0]
                rzs = rzs.exterior
            rz = np.array(rzs.coords).T
        else:
            rz = get_emc3_shape_simple(ds, direction, phi, coords, zsign)
            assert len(rz) == 1
            rz = rz[0]

        if return_early == 1:
            return rz

        def get_new_theta(rz, rev=False):
            drz = rz - np.roll(rz, 1, 1)
            ids = np.sqrt(drz[0] ** 2 + drz[1] ** 2)
            if rev:
                ids = 1 / np.array(ids)
            knew = np.cumsum(ids)
            knew *= np.pi * 2 / knew[-1]
            knew -= knew[0]
            return knew

        spl = []
        lk = len(rz[0])
        lke = lk + 10

        theta = get_new_theta(rz)
        theta = np.hstack((theta, theta[: lke - lk] + np.pi * 2))
        cutoff = np.pi * 2 / len(theta) / 100
        while any(theta[1:] - theta[:-1] <= cutoff):
            for i in range(len(theta) - 1):
                if theta[i + 1] - theta[i] <= cutoff:
                    try:
                        new = (theta[i] + theta[i + 2]) / 2
                    except IndexError:
                        new = (theta[i] + theta[-1] + np.pi * 2) / 2
                    theta[i + 1] = new

        k2 = np.linspace(0, np.pi * 2, nz, endpoint=False) + np.pi / lk * (lke - lk)
        splines = [
            make_smoothing_spline(theta, np.hstack((k, k[: lke - lk]))) for k in rz
        ]
        spl = [spl(k2) for spl in splines]

        assert_valid(np.array(spl))
        bounds = nz // 50
        bounds0 = bounds
        if bounds < 2:
            bounds = 2
        while bounds > 1:
            splnew = smooth(*spl, bounds=bounds, plot=0)
            bounds_smooth = bounds
            while True:
                try:
                    assert_valid(np.array(splnew))
                except:
                    if bounds_smooth < bounds0 + 1:
                        bounds_smooth += 1
                        splnew = smooth(
                            *spl, bounds=bounds_smooth, plot=0, bounds_find=bounds
                        )
                        continue
                    print(bounds, bounds_smooth, bounds0)
                    smooth(*spl, bounds=bounds_smooth, plot=True, bounds_find=bounds)
                    smooth(*spl, bounds=bounds, plot=True)
                    raise
                else:
                    break
            spl = splnew
            bounds //= 2
        points.append(spl)
    lines = [zb.rzline.RZline(*spl, spline_order=True) for spl in points]
    for line in lines:
        assert_valid(lines[-1])

    xr.DataArray(np.array(points), dims=("phi", "Rz", "theta")).to_netcdf(fnc)

    return lines


class timeit(object):
    def __init__(self, info="%f"):
        self.info = info

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        diff = time.time() - self.t0
        if args == (None, None, None):
            state = "succeeded"
        else:
            state = str(args)
        if "%s" in self.info:
            print(self.info % (state, diff))
        else:
            print(self.info % (diff))


class timeit2(timeit):
    def __init__(self, info):
        print(info, end="")
        sys.stdout.flush()
        self.info = f"\r{info} ... %s in %f s"


