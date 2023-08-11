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
from multiprocessing import Pool

no4 = np.array([0, 1, 2, 3, 5, 6, 7, 8])
nearlyeye = np.eye(9)[no4]
nearlyeye[:, 4] -= 1
sign0 = np.ones(9)
sign0[4] = -1
sign0 *= 1e4
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
    prog = tqdm(desc=f"Generating [{tot:2d}/{totals}]", total=n**2)
    for i in range(n):
        for j1, j2 in enumerate(j2s):
            A[..., i, j1] = taylor(x[i], np.array(j2)[slc])
            prog.update()
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


def errFunc(x0, B, target, fac):
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


v31 = [1, 4, 7]


def getv3(x):
    return np.array(
        (
            np.sum(x[1::3]) + 1,
            np.sum(x[3:6]) + 1,
            -(np.sum(x[::2] - np.sum(x[1::2]))) + 1,
            # -(np.sum(x[3:6]) - np.sum(x[:3]) - np.sum(x[6:])) + 0.1,
        )
    )


def errFunc2(x, B, target, sign):
    v2 = x * sign
    v2[v2 > 0] /= 1e6
    v3 = getv3(x)
    v3[v3 < 0] = 0
    return np.hstack((np.dot(x, B) - target, v2, v3))


v30 = np.zeros((9, 3))
v30[1::3, 0] = 1
v30[3:5, 1] = 1
v30[:, 2] = -1
v30[1::2, 2] = 1
# v30[:, 3] = 1
# v30[3:6, 3] = -1
v30 = v30.T


def jac2(x, B, t, sign):
    # v2 = sign * x
    v2 = sign.copy()
    v2[sign * x > 0] = 0  # /= 1e6
    v2 = np.diag(v2)
    v3 = v30.copy()
    v3[getv3(x) < 0] = 0
    return np.vstack((B.T, v2, v3))


testfuncs = {}
testfuncs["delp2_cartesian"] = [
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
# Jacobian is x
testfuncs["delp2"] = [
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
testfuncs["ddR"] = [
    (
        lambda x, y: np.sin(x),
        lambda x, y: np.cos(x),
    ),
    (
        lambda x, y: np.cos(x),
        lambda x, y: -np.sin(x),
    ),
    (
        lambda x, y: np.sin(y),
        lambda x, y: 0,
    ),
    (
        lambda x, y: np.sin(x + y),
        lambda x, y: np.cos(x + y),
    ),
    (
        lambda x, y: np.sin(x - y),
        lambda x, y: np.cos(x - y),
    ),
]
testfuncs["ddZ"] = [
    (
        lambda x, y: np.sin(y),
        lambda x, y: np.cos(y),
    ),
    (
        lambda x, y: np.cos(y),
        lambda x, y: -np.sin(y),
    ),
    (
        lambda x, y: np.sin(x),
        lambda x, y: 0,
    ),
    (
        lambda x, y: np.sin(x + y),
        lambda x, y: np.cos(x + y),
    ),
    (
        lambda x, y: np.sin(x - y),
        lambda x, y: -np.cos(x - y),
    ),
]


def read(fn):
    with DataFile(fn) as f:
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
    return genMatrix(dist)


_cache_u_f = ()


class SharedNumpy:
    __slots__ = (
        "arr",
        "shm",
        "name",
        "shared",
    )

    def __init__(self, arr: np.ndarray = None):
        if arr is not None:
            self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            self.arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
            self.name = self.shm.name
            np.copyto(self.arr, arr)

    def __getattr__(self, item):
        if hasattr(self.arr, item):
            return getattr(self.arr, item)
        raise AttributeError(
            f"{self.__class__.__name__}, doesn't have attribute {item!r}"
        )

    def __str__(self):
        return str(self.arr)

    @classmethod
    def from_name(cls, name, shape, dtype):
        memory = cls(arr=None)
        memory.shm = shared_memory.SharedMemory(name)
        memory.arr = np.ndarray(shape, dtype=dtype, buffer=memory.shm.buf)
        memory.name = name
        return memory

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def shape(self):
        return self.arr.shape


class LS:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        # self.old = SharedNumpy.from_name(self.old, self.shape, self.dtype)

    def getopt(s, ijk):
        ijk = tuple(ijk)
        return leastsq(
            s.ef,
            s.old[ijk],
            max_nfev=int(s.nfev),
            args=(s.A[ijk], s.target[ijk], s.epsn),
            gtol=s.tol,
            ftol=s.tol,
            xtol=s.tol,
            jac=s.ja,
        ).x


def update_facs(
    old, pos, A, target, testfunc, scale, scalepower, tol, eps, badeps=1e-2, nfev=1e3
):
    gridscale = (192 / A.shape[2]) ** scalepower
    badeps *= gridscale
    badeps *= scale
    eps /= gridscale
    tol *= gridscale
    tol = max(2.3e-16, tol)
    # print("starting dot")
    # t2 = np.empty((*A.shape[:3], A.shape[-1]))
    # for ijk in itertools.product(*[range(k) for k in A.shape[:-2]]):
    #     t2[ijk] = np.dot(old[ijk], A[ijk])  # .transpose(3, 0, 1, 2, 4))
    # print("finished dot")
    # plt.pcolormesh(*[x[:, 2] for x in pos], t2[:, 2, :, 3])
    # plt.colorbar()
    # plt.show()
    # errHist(getError(old, pos))
    # getBad(old, pos, 3)
    # getBad(old, pos, 3)
    # showMeBad(old, pos)
    # plt.show()
    nf = old.copy()
    bad = getBad(old, pos, testfunc, 0, cutoff=badeps)
    print(
        np.sum(getBad(nf, pos, testfunc, 0, badeps)),
        np.max(getError(nf, pos, testfunc)),
    )
    if not np.any(bad):
        return nf
    ijks = np.array(np.where(bad)).T

    print(ijks.shape)
    ef, ja, epsn = (
        (errFunc2, jac2, eps * sign0) if scalepower == 2 else (errFunc, jac, eps)
    )
    global tot, totals

    # ls = LS(ef=ef, old=old, nfev=nfev, A=A, target=target, epsn=epsn, tol=tol, ja=ja)

    results = tqdm(
        [
            pool.apply_async(
                leastsq,
                (ef, old[tuple(ijk)]),
                dict(
                    max_nfev=int(nfev),
                    args=(A[tuple(ijk)], target[tuple(ijk)], epsn),
                    gtol=tol,
                    ftol=tol,
                    xtol=tol,
                    jac=ja,
                ),
            )
            for ijk in ijks
        ],
        desc=f"Improving [{tot:2d}/{totals}]",
        smoothing=0,
    )

    for ijk, newfac in zip(ijks, results):
        nf[tuple(ijk)] = newfac.get().x

    # # def getopt(ef, old, nvew, A, target, epsn,  tol, ja, ijk)
    # with tqdm(
    #     desc=f"Improving [{tot:2d}/{totals}]", total=np.sum(np.array(bad)), smoothing=0
    # ) as prog:
    #     for ijk in itertools.product(*[range(k) for k in A.shape[:-2]], disable=True):
    #         if not bad[ijk]:
    #             continue
    #         newfac = leastsq(
    #             ef,
    #             old[ijk],
    #             max_nfev=int(nfev),
    #             args=(A[ijk], target[ijk], epsn),
    #             gtol=tol,
    #             ftol=tol,
    #             xtol=tol,
    #             jac=ja,
    #         )
    #         nf[ijk] = newfac.x
    #         prog.update()
    print(
        np.sum(getBad(nf, pos, testfunc, 0, badeps)),
        np.max(getError(nf, pos, testfunc)),
        tol,
        eps,
        badeps,
    )
    # errHist(getError(nf, pos))
    # getBad(old, pos, 3)
    # plt.show()
    return nf


def errHist(err):
    err = np.abs(err.flatten())
    space = np.geomspace(np.min(err[err > 0]), np.max(err))
    plt.figure()
    plt.hist(err, space)
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")


def getBad(nf, pos, tf, plot=1, cutoff=1e-2):
    return getError(nf, pos, tf, None, plot) > cutoff


def getError(nf, pos, testfuncs, slc=None, plot=0):
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


def showMeBad(nf, pos, tf):
    err = getError(nf, pos, tf)
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
    plt.pcolormesh(getFull(nf[ijk]).reshape((3, 3)))
    plt.colorbar()


def write(fn, nf, name):
    with DataFile(fn, write=True) as f:
        nff = getFull(nf)
        for i in range(nff.shape[-1]):
            f.write(
                f"{name}_3x3_{i}",
                xr.DataArray(nff[..., i], dims=("x", "y", "z")),
            )
        for j, d in enumerate("xz"):
            for i in range(nff.shape[-1]):
                f.write(f"{d}offset_3x3_{i}", posi[i][j])
    global tot
    tot += 1


posi = getPosi(1)
print(posi)
kws = getKW()


def doall(fn):
    global tot
    # tot = tqdm(total=len(kws) + 2, desc="Total progress", unit="step")
    pos = read(fn)
    A = invertMatrix(pos)
    tot += 1

    target = np.zeros(A.shape[:-1])
    target[..., 2] = 1
    target[..., 6] = 1
    target[..., 3] = 1 / pos[0]
    nf = doone(pos, A, target, testfuncs["delp2"], 10, 2)
    write(fn, nf, "delp2")

    target = np.zeros(A.shape[:-1])
    target[..., 3] = 1
    nf = doone(pos, A, target, testfuncs["ddR"], 1e-2, 1)
    write(fn, nf, "ddR")

    target = np.zeros(A.shape[:-1])
    target[..., 1] = 1
    nf = doone(pos, A, target, testfuncs["ddZ"], 1e-2, 1)
    write(fn, nf, "ddZ")


def doone(pos, A, target, testfunc, scale, scalepower):
    global tot

    delp2coefs = np.linalg.solve(A.transpose(0, 1, 2, 4, 3), target)

    nfs = [delp2coefs]

    for kw in kws:
        nfs.append(
            update_facs(nfs[-1], pos, A, target, testfunc, scale, scalepower, **kw)
        )
        tot += 1

    stats = []
    for i, nfi in enumerate(nfs):
        err = getError(nfi, pos, testfunc)
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
        getBad(nf, pos, testfunc, 3, 1e-1)
        errHist(getError(nf, pos, testfunc))
        showMeBad(nf, pos, testfunc)
        plt.show()
    return nf


pool = None
pool_size = 36

if __name__ == "__main__":
    totals = (len(kws) * 3 + 2) * len(sys.argv[1:])
    tot = 0
    with Pool(processes=pool_size) as pool:
        for fn in sys.argv[1:]:
            doall(fn)
# fn = "W7X-conf0-36x36x192.emc3.inner:f.vessel:f.island:f.fci.nc.new"
