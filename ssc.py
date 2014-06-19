import numpy as np
import scipy
import scipy.io
import tables

_ssc_random_state = np.random.RandomState(0)

#def is_vector(x):
#    return np.ndarray == x.__class__ and len(x.shape) == 1

class ImageData:
    def __init__(self, images_file="IMAGES.mat", patch_width=8,
               patch_height=8, random_state=_ssc_random_state):
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
    def __init__(self, nu=0.02, nu_homeo=0.01, n_quant=256):
        self.nu = nu
        self.nu_homeo = nu_homeo
        self.n_quant = n_quant
        self.edges = np.linspace(0, 1., self.n_quant+1)

    def init(self):
        self.data_size = self.psi.shape[0]
        self.num_basis = self.psi.shape[1]
        self.size = self.psi.shape[0]
        self.grad = np.outer(np.ones((self.psi.shape[1])), self.edges[:-1])
        self.shifter = np.arange(self.num_basis)*(self.n_quant)
        self.psi /= np.sqrt(np.sum(self.psi ** 2, axis=0))

    def save_hdf5(self, filename):
        h5file = tables.openFile(filename, mode="w", title="SSC coder basis and learning parameters")
        group = h5file.createGroup(h5file.root, "Coder", "The coder state")
        h5file.createArray(group, 'psi', self.psi, "Basis matrix")
        h5file.createArray(group, 'f', self.f, "Basis gain")
        h5file.createArray(group, 'i_iter', self.i_iter, "learning step")
        h5file.createArray(group, 'iters', self.iters, "total learning steps")
        h5file.createArray(group, 'L0', self.L0, "L0 norm")
        h5file.createArray(group, 'SE', self.SE, "residual energy")
        h5file.close()


    def load_hdf5(self, filename, iters=0):
        h5file = tables.openFile(filename, 'r')
        self.psi = np.array(h5file.getNode('/Coder', 'psi').read())
        self.f = np.array(h5file.getNode('/Coder', 'f').read())
        self.i_iter = np.array(h5file.getNode('/Coder', 'i_iter').read())
        self.iters = np.array(h5file.getNode('/Coder', 'iters').read())
        self.L0 = np.array(h5file.getNode('/Coder', 'L0').read())
        self.SE = np.array(h5file.getNode('/Coder', 'SE').read())
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
        # self.psi = random_state.standard_normal((data_size, num_basis))
        self.psi = random_state.lognormal(size=(data_size, num_basis))
        self.num_basis = self.psi.shape[1]
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
        threshold = 0.1 * np.dot(e.T, e)
        steps, steps_max = 0, z.size # len(z)
        while (np.dot(e.T, e) > threshold) and (steps < steps_max):
            # Matching
            # non-negative coefficients (does not assume ON-OFF symmetry of RFs)
#             z = self.f.ravel()[self.shifter + np.int16((c > 0)*c*self.n_quant)]
            # absolute  coefficients (does assume ON-OFF symmetry of RFs as in SparseNet)
            z = self.f.ravel()[self.shifter + np.int16(np.absolute(c)*self.n_quant)]
            ind = z.argmax()
            a[ind] = c[ind]
            # Pursuit
            e -= c[ind]*self.psi[:, ind]
            c = np.dot(e.T, self.psi) # activities
            steps += 1
            # TODO: use self.C to update activities
        return a

    def update(self, x, a):
        self.f *= (1 - self.nu_homeo)
        self.f += self.nu_homeo * (self.grad > np.absolute(a)[:, np.newaxis])

        residual = x - np.dot(self.psi, a)
        self.psi += self.nu * np.outer(residual, a)
        self.psi /= np.sqrt(np.sum(self.psi ** 2, axis=0))
