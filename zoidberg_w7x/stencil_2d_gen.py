#!/usr/bin/env python
# coding: utf-8

# In[170]:

import sys
import inspect
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from boututils.datafile import DataFile
from scipy.optimize import least_squares as leastsq
from tqdm.auto import tqdm
from tqdm.contrib import itertools

no4 = np.array([0, 1, 2, 3, 5, 6, 7, 8])
nearlyeye = np.eye(9)[no4]
nearlyeye[:, 4] -= 1

tot = 0
totals = 0


def getKW():
    kw = [
        dict(tol=1e-9, eps=1e-9, badeps=1e-5),
        dict(tol=1e-10, eps=1e-8, badeps=1e-2),
        dict(tol=1e-12, eps=1e-6, badeps=1e-3),
        dict(tol=1e-12, eps=1e-7, badeps=1e-3),
    ]

    for fac in (1e-7, 1e-8, 1e-9):
        kw.append(dict(tol=1e-12, eps=fac, badeps=1e-3, nfev=1e4))

    for fac in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        kw.append(dict(tol=1e-13, eps=fac, badeps=5e-3, nfev=1e4))

    for fac in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9)[::-1]:
        kw.append(dict(tol=1e-13, eps=fac, badeps=10e-3, nfev=1e4))

    for fac in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9)[::-1]:
        kw.append(dict(tol=1e-13, eps=fac, badeps=1e-3, nfev=1e4))

    return kw


def factorial(a):
    if isinstance(a, np.ndarray):
        return np.array([factorial(b) for b in a])
    if a == 0 or a == 1:
        return 1
    else:
        assert a > 0
        return a * factorial(a - 1)


def taylor(x, j):
    if np.all(j) >= 0:
        return np.prod(x**j / factorial(j), axis=0)
    else:
        return 0


def genMatrix(x):
    n = len(x)
    shape = x.shape[2:]
    A = np.zeros((*shape, n, n))
    n2 = int(np.sqrt(n))
    slc = tuple([slice(None)] + [None for _ in shape])

    j2s = list(itertools.product(range(n2), range(n2), disable=True))
    print(j2s)
    for i in range(n):
        for j1, j2 in enumerate(j2s):
            A[..., i, j1] = taylor(x[i], np.array(j2)[slc])
    return A


def test(R, Z, inp, ana, coef, posis, show=False, quiet=False):
    args = inp, ana
    inp = inp(R, Z)
    ana = ana(R, Z)
    comp = 0

    def roll(f, a, b):
        return np.roll(np.roll(f, -a, 0), -b, 2)

    for i, posi in enumerate(posis):
        comp += roll(inp, *posi) * coef[..., i]
    slc = (slice(1, -1), 1)

    err = comp - ana

    def argmax(a, axis=None):
        return np.unravel_index(np.argmax(a, axis=axis), a.shape)

    ijk = argmax(np.abs(err)[1:-1])
    if not quiet:
        print(ijk)
        print(np.sqrt(np.mean(((comp - ana)[1:-1]) ** 2)))
    slc = (slice(1, -1), ijk[1])
    # return
    err = comp - ana
    if not show:
        return err
    import matplotlib.pyplot as plt

    if show == 4:
        plt.figure()
        plt.scatter(R[1:-1], err[1:-1], alpha=0.3, c=Z[1:-1])
        pa, pb = np.percentile(err[1, -1], [1, 99])
        pm = (pa + pb) / 2
        pd = (pb - pa) * 2
        plt.ylim(pm - pd, pm + pd)
        plt.title("Error of " + inspect.getsource(args[0]))
        return err

    def myplot(d):
        plt.figure()
        kw = dict()
        if show > 2:
            kw["vmin"], kw["vmax"] = np.percentile(d[slc], [1, 99])
        plt.pcolormesh(R[slc], Z[slc], d[slc], **kw)
        plt.colorbar()

    myplot(err)
    plt.title("Error of " + inspect.getsource(args[0]))
    if show < 2:
        plt.show()
        return err

    myplot(comp)
    plt.title("Numerical of " + inspect.getsource(args[0]))
    myplot(ana)
    plt.title("Analytical = " + inspect.getsource(args[1]))
    plt.show()
    return err


def getFull(x0, axis=0):
    return x0
    if axis == 0:
        return np.array((*x0[:4], -np.sum(x0), *x0[4:]))
    slc = [slice(None) for _ in range(axis + 1)]
    slc[axis] = slice(None, 4)
    tmp = [x0[tuple(slc)]]
    slc[axis] = None
    tmp += [-np.sum(x0, axis=axis)[tuple(slc)]]
    slc[axis] = slice(4, None)
    tmp += [x0[tuple(slc)]]
    return np.concatenate(tmp, axis=axis)


def errfunc(x0, B, target, fac):
    x = getFull(x0)
    return np.hstack((np.dot(x, B) - target, x * fac))


def jac(x, B, t, fac):
    return np.vstack((B.T, fac * np.eye(9)))
    Bd = B[no4] - B[4:5]
    return np.vstack((Bd.T, fac * nearlyeye.T))


def argmax(a, axis=None):
    return np.unravel_index(np.argmax(a, axis=axis), a.shape)


def getSlc(ijk):
    ijkp = [slice(i - 1, i + 2) for i in ijk]
    ijkp[1] = ijk[1]
    return tuple(ijkp)


def Stencil(facs, ijk, func):
    ijkp = getSlc(ijk)
    return np.sum(func[ijkp] * facs.reshape(3, 3))


def errFunc2(x, B, target, sign):
    v2 = x * sign
    v2[v2 > 0] = 0
    return np.hstack((np.dot(x, B) - target, v2))


def jac2(x, B, t, sign):
    v2 = sign * x
    v2 = sign.copy()
    v2[sign * x > 0] = 0
    v2 = np.diag(v2)
    return np.vstack((B.T, v2))


cartesian = False
if cartesian:
    # Jacobian is unitiy
    testfuncs = [
        (
            lambda x, y: np.sin(x),
            lambda x, y: np.sin(-x),
        ),
        (
            lambda x, y: np.sin(y),
            lambda x, y: np.sin(-y),
        ),
        (
            lambda x, y: np.sin(x + y),
            lambda x, y: 2 * np.sin(-x - y),
        ),
        (
            lambda x, y: np.sin(x - y),
            lambda x, y: 2 * np.sin(-x + y),
        ),
    ]
else:
    # Jacobian is x
    testfuncs = [
        (
            lambda x, y: np.sin(x),
            lambda x, y: -np.sin(x) + np.cos(x) / x,
        ),
        (
            lambda x, y: np.cos(x),
            lambda x, y: -np.cos(x) - np.sin(x) / x,
        ),
        (
            lambda x, y: np.sin(y),
            lambda x, y: np.sin(-y),
        ),
        (
            lambda x, y: np.sin(x + y),
            lambda x, y: -2 * np.sin(x + y) + np.cos(x + y) / x,
        ),
        (
            lambda x, y: np.sin(x - y),
            lambda x, y: -2 * np.sin(x - y) + np.cos(x - y) / x,
        ),
    ]


def read(fn):
    with DataFile(fn, write=True) as f:
        return f["R"], f["Z"]


def getPosi(n):
    posi1d = range(-n, n + 1)
    posi = np.array(
        [(i, j) for i, j in itertools.product(posi1d, posi1d, disable=True)]
    )
    return posi


def invertMatrix(pos):
    # pos = [x[::2, ::2, :2] for x in pos]
    # pos = [x[:, :1, :] for x in pos]
    dist = -np.array(
        [[x - np.roll(np.roll(x, -i, 0), -j, 2) for x in pos] for i, j in posi]
    )
    A = genMatrix(dist)
    iA = np.empty_like(A)
    iA[:] = np.nan
    for ijk in itertools.product(
        *[range(k) for k in A.shape[:-2]], desc=f"Inverting [{tot:2d}/{totals}]"
    ):
        iA[ijk] = np.transpose(np.linalg.inv(np.transpose(A[ijk])))
    return A, iA


def update_facs(old, pos, A, tol, eps, badeps=1e-2, nfev=1e3):
    iR = 1 / pos[0]
    # print("starting dot")
    # t2 = np.empty((*A.shape[:3], A.shape[-1]))
    # for ijk in itertools.product(*[range(k) for k in A.shape[:-2]]):
    #     t2[ijk] = np.dot(old[ijk], A[ijk])  # .transpose(3, 0, 1, 2, 4))
    # print("finished dot")
    # plt.pcolormesh(*[x[:, 2] for x in pos], t2[:, 2, :, 3])
    # plt.colorbar()
    # plt.show()
    errHist(getError(old, pos))
    # getBad(old, pos, 3)
    getBad(old, pos, 3)
    showMeBad(old, pos)
    plt.show()
    nf = old.copy()
    # target = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0])
    bad = getBad(old, pos, 0, cutoff=badeps)
    print(np.sum(getBad(nf, pos, 0, badeps)), np.max(getError(nf, pos)))
    global tot, totals
    with tqdm(
        desc=f"Improving [{tot:2d}/{totals}]", total=np.sum(np.array(bad)), smoothing=0
    ) as prog:
        for ijk in itertools.product(*[range(k) for k in A.shape[:-2]], disable=True):
            if not bad[ijk]:
                continue
            target = np.array([0, 0, 1, iR[ijk], 0, 0, 1, 0, 0])
            newfac = leastsq(
                errfunc,
                old[ijk],
                max_nfev=int(nfev),
                args=(A[ijk], target, eps),
                gtol=tol,
                ftol=tol,
                xtol=tol,
                jac=jac,
            )
            nf[ijk] = newfac.x
            prog.update()
    print(np.sum(getBad(nf, pos, 0, badeps)), np.max(getError(nf, pos)))
    errHist(getError(nf, pos))
    getBad(old, pos, 3)
    plt.show()
    return nf


def write(fn, nf, posi):
    with DataFile(fn, write=True) as f:
        f.write(
            "delp2_3x3",
            xr.DataArray(nf, dims=("x", "y", "z", "offset_3x3")),
        )
        f.write(
            "offset_3x3",
            xr.DataArray(posi, dims=("offset", "delta_xz")),
        )


def errHist(err):
    err = np.abs(err.flatten())
    space = np.geomspace(np.min(err[err > 0]), np.max(err))
    plt.figure()
    plt.hist(err, space)
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")


def getBad(nf, pos, plot=1, cutoff=1e-2):
    return getError(nf, pos, None, plot) > cutoff


def getError(nf, pos, slc=None, plot=0):
    err = None
    if slc is None:
        slc = slice(None)
    assert isinstance(slc, slice)
    for func in testfuncs[slc]:
        erri = test(*pos, *func, getFull(nf, 3), posi, plot, 1)
        erri = np.abs(erri)
        erri[0] = 0
        erri[-1] = 0
        if err is None:
            err = erri
        else:
            err = np.max((err, erri), axis=0)
            assert err.shape == erri.shape
    return err


def showMeBad(nf, pos):
    err = getError(nf, pos)
    ijk = argmax(err)
    j = ijk[1]
    slc = slice(1, -1), j
    R, Z = pos
    plt.figure()
    plt.pcolormesh(*[x[slc] for x in pos], err[slc])
    t = np.linspace(0, 2 * np.pi)
    f = 0.05
    plt.plot(*[x[ijk] + f * fun(t) for x, fun in zip(pos, (np.sin, np.cos))])
    slc = getSlc(ijk)
    plt.plot(*[x[slc] for x in pos])
    plt.plot(*[x[slc].T for x in pos])
    plt.colorbar()
    plt.figure()
    plt.pcolormesh(nf[ijk].reshape((3, 3)))
    plt.colorbar()


posi = getPosi(1)
print(posi)
kws = getKW()


def doall(fn):
    # tot = tqdm(total=len(kws) + 2, desc="Total progress", unit="step")
    global tot

    pos = read(fn)
    A, iA = invertMatrix(pos)
    tot += 1
    delp2coefs = iA[..., 2, :] + iA[..., 6, :] + iA[..., 3, :] / pos[0][..., None]
    # delp2coefs = delp2coefs[:, :, :, no4]

    nfs = [delp2coefs]

    for kw in kws:
        nfs.append(update_facs(nfs[-1], pos, A, **kw))
        tot += 1

    stats = []
    for i, nfi in enumerate(nfs):
        err = getError(nfi, pos)
        stats.append((i, np.max(err), np.mean(err)))

    bests = []
    for key in (lambda x: (x[1] + x[2]), lambda x: (x[2]), lambda x: (x[1])):
        for i in sorted(stats, key=key)[:3]:
            bests.append(i)
            print(i)
        print()
    print(bests[::3])
    best = bests[0]
    nf = nfs[best[0]]
    if best[2] > 1e-1:
        getBad(nf, pos, 3)
        errHist(getError(nf, pos))
        showMeBad(nf)
        plt.show()

    with DataFile(fn, write=True) as f:
        nff = np.concatenate(nf[..., :4], -np.sum(nf, axis=-1), nf[..., 4:])
        f.write(
            "delp2_3x3",
            xr.DataArray(nff, dims=("x", "y", "z", "offset_3x3")),
        )
        f.write(
            "offset_3x3",
            xr.DataArray(posi, dims=("offset", "delta_xz")),
        )
        tot += 1


if __name__ == "__main__":
    totals = (len(kws) + 2) * len(sys.argv[1:])
    tot = 0
    for fn in sys.argv[1:]:
        doall(fn)
# fn = "W7X-conf0-36x36x192.emc3.inner:f.vessel:f.island:f.fci.nc.new"
