#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import ssc
try:
    import progressbar
    PROGRESS = True
except:
    print 'progressbar could not be imported'
    PROGRESS = False

iters = 50000
each_iter = iters/1000 # how often we refresh plotting / saving data

#databases = ["sparsenet", "icabench_decorr"]
databases = ["sparsenet"]
import os
def show_basis(psi):
    def get_format(count):
        rows = np.ceil(np.sqrt(count))
        cols = np.ceil(count / rows)
        return int(rows), int(cols)

    fig = plt.figure(figsize=(12, 12), subplotpars=matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0.05, hspace=0.05,))
    rows, cols = get_format(psi.shape[1])
    for col in range(psi.shape[1]):
#    a = fig.add_axes((border, border, 1.-2*border, 1.-2*border))#, frameon=False, axisbg='w')
        a = fig.add_subplot(rows, cols, col + 1)
    # Assume square patch. Height/width order are probably wrong.
        height = width = int(np.sqrt(psi.shape[0]))
        a.imshow(np.reshape(psi[:, col], (height, width)),
                 interpolation="nearest", cmap=plt.cm.gray)
        a.axis("off")
    return fig


def ssc_learn(images_file="data/IMAGES.mat", iters=100,
              patch_width = 10, patch_height = 10, num_basis = 196,
              load_file="data/ssc_test.hdf5", save_file="data/ssc_test.hdf5"):

    image_data = ssc.ImageData(images_file=images_file, patch_width=patch_width, patch_height=patch_height)

    coder = ssc.Ssc()
    if not os.path.exists(load_file):
        coder.init_random(patch_width * patch_height, num_basis, iters=iters)
    else:
        coder.load_hdf5(load_file, iters=iters)

    if PROGRESS:
        pbar = progressbar.ProgressBar(widgets=["calculating", " ", progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ', progressbar.ETA()], maxval=coder.iters)
        pbar.start()
        pbar.update(coder.i_iter)

    while (coder.i_iter < coder.iters):

        x = image_data.draw()
        a = coder.sparsify(x)
        coder.update(x, a)

        # stats
        residual = x - np.dot(coder.psi, a)
        coder.L0[coder.i_iter], coder.SE[coder.i_iter] = np.nonzero(a)[0].size*1./num_basis, np.dot(residual.T, residual)
        if (coder.i_iter % each_iter) == 0:
            fig = plt.figure(figsize=(6, 6))
            a = fig.add_subplot(111)
            a.plot(coder.L0, 'b', alpha=.5)
            a.plot(coder.SE, 'r', alpha=.5)
#            a.plot(1-coder.L0, coder.SE, alpha=.1)
            fig.savefig(images_file.replace('data', 'results') + '_L0vsSE.pdf')
            fig = show_basis(coder.psi)
            fig.savefig(images_file.replace('data', 'results') + '.pdf')
            plt.close('all')
            coder.save_hdf5(save_file)

        coder.i_iter += 1
        if PROGRESS: pbar.update(coder.i_iter)
    if PROGRESS: pbar.finish()

if __name__ == "__main__":
    for name in databases:
        matfile = "data/IMAGES_" + name + ".mat"
        print "learning with the ", name, " database "
        ssc_learn(images_file=matfile, iters=iters,
                  load_file="data/" + name + ".hdf5",
                  save_file="data/" + name + ".hdf5")

