import numpy as np
import scipy
import scipy.io
import tables

_ssc_random_state = np.random.RandomState(0)

class ImageData:
    def __init__(self, images_file="data/IMAGES.mat", patch_width=8,
               patch_height=8, random_state=_ssc_random_state):
        import os
        if not os.path.exists(images_file):
            print "Downloading data ", images_file
            import urllib
            URL = 'http://invibe.net/LaurentPerrinet/SparseHebbianLearning?action=AttachFile&do=get&target='
            opener = urllib.urlopen(URL + images_file.replace('data/', ''))
            open(matfile, 'wb').write(opener.read())

        # Image array is in order x, y, n.
        self.images = scipy.io.loadmat(images_file)["IMAGES"]
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rs = random_state

    def draw(self):
        n = self.rs.randint(self.images.shape[2])
        c = self.rs.randint(self.images.shape[0] - self.patch_width)
        r = self.rs.randint(self.images.shape[1] - self.patch_height)
        patch = self.images[:, :, n].\
            take(c + np.arange(self.patch_width), axis=0).\
            take(r + np.arange(self.patch_height), axis=1)
        x = np.reshape(patch, (-1,))
        x -= np.mean(x)
        x /= np.sqrt(np.sum(x ** 2))
        return x

class Coder:
    pass

class Ssc(Coder):
    def __init__(self, nu=0.02, nu_homeo=0.01, alpha=0.02, threshold=0.01):
        self.nu = nu
        self.nu_homeo = nu_homeo
        self.alpha = alpha
        self.n_quant = 256 # should not change if we use the shifter trick
        self.edges = 0.1*np.linspace(0, 1., self.n_quant+1)
        self.threshold = threshold

    def init(self):
        self.data_size = self.psi.shape[0]
        self.num_basis = self.psi.shape[1]
        self.size = self.psi.shape[0]
        self.grad = np.outer(np.ones((self.psi.shape[1])), self.edges[:-1])
        self.shifter = np.arange(self.num_basis)*(self.n_quant)
        self.psi /= np.sqrt(np.sum(self.psi ** 2, axis=0))

    def save_hdf5(self, filename):
        h5file = tables.open_file(filename, mode="w", title="SSC coder basis and learning parameters")
        group = h5file.create_group(h5file.root, "Coder", "The coder state")
        h5file.create_array(group, 'psi', self.psi, "Basis matrix")
        h5file.create_array(group, 'f', self.f, "Basis gain")
        h5file.create_array(group, 'i_iter', self.i_iter, "learning step")
        h5file.create_array(group, 'iters', self.iters, "total learning steps")
        h5file.create_array(group, 'L0', self.L0, "L0 norm")
        h5file.create_array(group, 'SE', self.SE, "residual energy")
        h5file.close()


    def load_hdf5(self, filename, iters=0):
        h5file = tables.openFile(filename, 'r')
        self.psi = np.array(h5file.get_node('/Coder', 'psi').read())
        self.f = np.array(h5file.get_node('/Coder', 'f').read())
        self.i_iter = np.array(h5file.get_node('/Coder', 'i_iter').read())
        self.iters = np.array(h5file.get_node('/Coder', 'iters').read())
        self.L0 = np.array(h5file.get_node('/Coder', 'L0').read())
        self.SE = np.array(h5file.get_node('/Coder', 'SE').read())
        if iters > self.iters:
            print self.L0.shape, np.ones((iters-self.iters)).shape
            self.L0 = np.hstack((self.L0, np.ones((iters-self.iters))))
            self.SE = np.hstack((self.SE, np.zeros((iters-self.iters))))
            self.iters = iters
        elif iters < self.iters:
            self.L0 = self.L0[:iters]
            self.SE = self.SE[:iters]
            self.iters = iters
        h5file.close()

        self.init()

    def init_random(self, data_size, num_basis, iters,
                  random_state=_ssc_random_state):
        self.psi = random_state.standard_normal((data_size, num_basis))
#         self.psi = random_state.lognormal(size=(data_size, num_basis))
        self.num_basis = self.psi.shape[1]
        self.S_var = 0.1* np.ones(self.num_basis)
        self.gain = np.ones(self.num_basis)
        self.f = np.outer(np.ones((self.num_basis)), self.edges[:-1])
        self.i_iter = 0
        self.iters = iters
        self.L0 = np.ones(self.iters)
        self.SE = np.zeros(self.iters)
        self.init()

    def sparsify(self, x):
        assert x.size == self.data_size
        c = np.dot(x.T, self.psi) # activities
        z = np.zeros_like(c)
        a = np.zeros_like(c)  # output sparse vector
        e = x.copy() # residual
        threshold = self.threshold * np.dot(e.T, e)
        steps, steps_max = 0, z.size
        while (np.dot(e.T, e) > threshold) and (steps < steps_max):
            # Matching
            # absolute  coefficients (does assume ON-OFF symmetry of RFs as in SparseNet)
            for i in range(self.num_basis):
                #      np.interp(x,                 xp,              fp, left=None, right=None)
                z[i] = np.interp(np.absolute(c[i]), self.edges[:-1], self.f[i, :], left=0, right=1)
            ind = z.argmax()
#             z = self.f.ravel()[self.shifter + np.int16(np.absolute(c)*self.n_quant)]
#             print z.argmax(), ind
#             ind = np.abs(c).argmax()
            a[ind] = c[ind]
            # Pursuit
            e -= c[ind]*self.psi[:, ind]
            c = np.dot(e.T, self.psi) # activities
            steps += 1
            # TODO: use self.C to update activities
        return a

    def update(self, x, a):
        residual = x - np.dot(self.psi, a)
        self.psi += self.nu * np.outer(residual, a)

        self.f *= (1 - self.nu_homeo)
        self.f += self.nu_homeo * (self.grad > np.absolute(a)[:, np.newaxis])

        self.S_var = (1-self.nu_homeo)*self.S_var + self.nu_homeo*(a**2)
#         self.gain *= (self.S_var)**self.alpha
#         self.psi *= self.gain
#         print self.gain,  self.S_var/self.S_var.mean(), np.sqrt(np.sum(self.psi ** 2, axis=0))
#         print self.S_var, self.S_var.mean()
        self.psi /= np.sqrt(np.sum(self.psi ** 2, axis=0))

