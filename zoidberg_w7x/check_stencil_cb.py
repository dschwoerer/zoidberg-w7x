from boututils.datafile import DataFile as DF
import sys
import numpy as np
import matplotlib.pyplot as plt


def read(fn, k="delp2", w="3x3"):
    with DF(fn) as f:
        dat = [f[f"{k}_{w}_{i}"] for i in range(9)]
        return np.array(dat)


def readposi(fn, w="3x3"):
    out = []
    with DF(fn) as f:
        for d in "xz":
            out.append([f[f"{d}offset_{w}_{i}"] for i in range(9)])
    return np.array(out)


def cb1(shape):
    cb = np.ones(shape)
    cb[::2] *= -1
    cb[:, ::2] *= -1
    cb[:, :, ::2] *= -1
    return cb


def cb2(shape):
    cb = np.ones(shape)
    cb[::2] *= -1
    return cb


def cb3(shape):
    cb = np.ones(shape)
    cb[:, :, ::2] *= -1
    return cb


def comp(inp, posis, coef):
    comp = 0

    def roll(f, a, b):
        return np.roll(np.roll(f, -a, 0), -b, 2)

    for i, posi in enumerate(posis.T):
        comp += roll(inp, *posi) * coef[i]
    comp[0] = 0
    comp[-1] = 0
    return comp


def argmax(d):
    return np.unravel_index(np.argmax(d), d.shape)


def pos(fn):
    with DF(fn) as f:
        return [f[x] for x in "RZ"]


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        coef = read(fn)
        posi = readposi(fn)
        RZ = pos(fn)
        econs = np.sum(coef, axis=0)
        _, j, _ = argmax(np.abs(econs))
        for j in j, 32:
            plt.figure()
            plt.pcolormesh(*[x[:, j] for x in RZ], econs[:, j])
            plt.colorbar()
        for cb in cb1, cb2, cb3:
            cb = cb(coef[0].shape)
            res = cb * comp(cb, posi, coef)
            _, j1, _ = argmax(res)
            ijk = argmax(res)
            print(argmax(res))
            print(np.max(res))
            print(np.min(res))
            print(np.sum(res > 0, axis=(0, 2)), res.size)
            plt.figure()
            plt.imshow(coef[(slice(None), *ijk)].reshape(3, 3))
            plt.colorbar()
            j2 = np.argmax(np.sum(res > 0, axis=(0, 2)))
            res[res < 0] = 0
            for j in j1, j2:
                # plt.figure()
                f, axs = plt.subplots(3)
                for ax, d in zip(axs, [cb, comp(cb, posi, coef), res]):
                    pl = ax.pcolormesh(*[x[:, j] for x in RZ], d[:, j])
                    plt.colorbar(pl, ax=ax)
                    # axs[1].pcolormesh(*[x[:, j] for x in RZ], comp(cb, posi, coef)[:, j])
                    # axs[2].pcolormesh(*[x[:, j] for x in RZ], res[:, j])
                    # plt.colorbar()
        plt.show()
