import os, sys, numpy, scipy, tempfile
from pyscf import lib

def get_h1e_tb(nsite, nmode, is_pbc=True):
    """
    Generate the one-electron Hamiltonian matrix for a tight-binding model.

    Parameters:
    - nsite (int): Number of sites in the lattice.
    - nmode (int): Number of modes (not used in this function but included for consistency).
    - is_pbc (bool): Whether to use periodic boundary conditions.

    Returns:
    - h1e (ndarray): The one-electron Hamiltonian matrix.
    """
    # Initialize the Hamiltonian matrix with zeros
    h1e = numpy.zeros((nsite, nsite))

    # Fill in the off-diagonal terms for nearest neighbors
    for i in range(nsite-1):
        h1e[i, i+1] = -1.0
        h1e[i+1, i] = -1.0

    # If periodic boundary conditions are used, connect the last site to the first site
    if is_pbc:
        h1e[0, nsite-1] = -1.0
        h1e[nsite-1, 0] = -1.0

    return h1e

def get_h1p1e_hol(nsite, nmode, hol_g=1.0):
    """
    Generate the one-electron and one-phonon Hamiltonian matrix with Holstein coupling.

    Parameters:
    - nsite (int): Number of sites.
    - nmode (int): Number of phonon modes.
    - hol_g (float): Holstein coupling constant.

    Returns:
    - h1p1e (ndarray): The one-electron and one-phonon Hamiltonian matrix.
    """
    # Initialize the Hamiltonian matrix with zeros
    h1p1e = numpy.zeros((nmode, nsite, nsite))

    # Fill in the diagonal terms for Holstein coupling
    for alph in range(nmode):
        i = alph
        h1p1e[alph, i, i] = hol_g

    return h1p1e

def get_h1p_hol(nsite, nmode, omega=1.0):
    """
    Generate the one-phonon Hamiltonian matrix for a Holstein model.

    Parameters:
    - nsite (int): Number of sites (not used in this function but included for consistency).
    - nmode (int): Number of phonon modes.
    - omega (float): Phonon frequency.

    Returns:
    - h1p (ndarray): The one-phonon Hamiltonian matrix.
    """
    # Initialize the Hamiltonian matrix with zeros
    h1p = numpy.zeros((nmode, nmode))

    # Fill in the diagonal terms with phonon frequencies
    for alph in range(nmode):
        h1p[alph, alph] = omega

    return h1p

def solve_hf(hol_obj, coeff0=None, verbose=0):
    log = lib.logger.new_logger(hol_obj, hol_obj.verbose)
    log.debug("\n\nSolving the variational polaron problem.")

    nsite = hol_obj.nsite
    nmode = hol_obj.nmode

    assert isinstance(hol_obj, NoSpin1eHolsteinModel)
    assert hol_obj.nelec_alph == 1
    assert hol_obj.nelec_beta == 0

    coeff_e = coeff0
    if coeff_e is None:
        coeff_e = numpy.eye(nsite)

    norb = nsite
    nocc = hol_obj.nelec_alph + hol_obj.nelec_beta
    nvir = norb - nocc

    h1e = hol_obj.get_h1e()
    assert h1e.shape == (nsite, nsite)

    h1p1e = hol_obj.get_h1p1e()
    assert h1p1e.shape == (nmode, nsite, nsite)

    hpp = hol_obj.get_h1p()
    assert hpp.shape == (nmode, nmode)
    
    f1e    = h1e
    ene_mo = numpy.einsum("mi,ni,mn->i", coeff_e, coeff_e, f1e, optimize=True)

    ene_cur = None
    ene_pre = None
    ene_err = 1.0

    icycle  = 0
    is_converged = False
    is_max_cycle = False

    orbv = None
    orbo = None
    rho  = None

    # Calculate the error every two iterations
    while not (is_converged or is_max_cycle):
        occidx, viridx = (lambda m: (m[:nocc], m[nocc:]))(numpy.argsort(ene_mo))

        # Always rearrange the orbitals to make sure the zero-th orbital
        # is the one with larges occupation number
        orbo = coeff_e[:, occidx].reshape(nsite, nocc)
        orbv = coeff_e[:, viridx].reshape(nsite, nvir)
        rho = numpy.einsum("mi,ni->mn", orbo, orbo)

        xi_rhs = numpy.einsum('Imn,mn->I', h1p1e, rho)
        xi     = numpy.linalg.solve(hpp, xi_rhs)

        f1e = h1e - numpy.einsum("Imn,I->mn", h1p1e, xi, optimize=True) * 2.0
        ene_mo, coeff_e = scipy.linalg.eigh(f1e)

        ene_pre  = ene_cur
        ene_cur  = numpy.einsum("mn,mn->", rho, h1e)
        ene_cur -= numpy.einsum("IJ,I,J->", hpp, xi, xi)

        if ene_pre is not None:
            ene_err = numpy.abs(ene_cur - ene_pre)
            log.debug('\niter %4d, energy = %12.8f, energy error = %6.4e',
                       icycle, ene_cur, ene_err)
            
            log.debug('xi = [%s]', ", ".join(["% 6.4f" % x for x in xi]))
            log.debug('n  = [%s]', ", ".join(["% 6.4f" % n for n in rho.diagonal()]))
            log.debug('e  = [%s]', ", ".join(["% 6.4f" % e for e in ene_mo]))
            # numpy.savetxt(log.stdout, coeff_e, fmt="% 6.4f", delimiter=", ")

        is_converged = ene_err < min(hol_obj.tol, 1e-8)
        is_max_cycle = icycle  > max(hol_obj.max_cycle, 500)
        icycle += 1

    if not is_converged:
        log.warn('f1e did not converge')

    assert orbo.shape == (nsite, nocc)
    assert orbv.shape == (nsite, nvir)
    assert rho.shape  == (nsite, nsite)

    hpx  = numpy.einsum("IJ,J->I", hpp, -xi)
    hpx += numpy.einsum("Imn,mn->I", h1p1e, rho, optimize=True)

    class VariationalPolaronResult(object):
        pass

    res = VariationalPolaronResult()
    res.ene     = ene_cur
    res.const   = numpy.einsum("IJ,I,J->", hpp, xi, xi)
    res.nsite = nsite
    res.nmode = nmode
    res.nph   = hol_obj.nph
    res.nelec_alph = hol_obj.nelec_alph
    res.nelec_beta = hol_obj.nelec_beta
    res.tmpdir = hol_obj.tmpdir

    res.coeff_e = coeff_e
    res.orbo    = orbo
    res.orbv    = orbv
    res.rho     = rho

    res.f1e   = f1e
    res.g1p1e = h1p1e

    res.h1e   = h1e
    res.h1p1e = h1p1e

    res.hpp   = hpp
    res.hpx   = hpx
    res.xi    = xi

    return res

class NoSpin1eHolsteinModel(lib.StreamObject):
    tmpdir   = os.environ.get('PYSCF_TMPDIR', lib.param.TMPDIR)
    _chkfile = None
    chkfile  = None

    is_pbc = True       # Periodic boundary condition flag
    is_cc_shift = True  # Whether to shift the harmonic oscillator basis
    is_fci_shift = True # Whether to shift the harmonic oscillator basis

    nsite  = 4         # Number of sites in the lattice
    nmode  = None      # Number of modes (will be set to 'nsite' in the constructor)
    nph    = 10    # Maximum number of phonons

    nelec_alph = 1     # Number of alpha electrons
    nelec_beta = 0     # Number of beta electrons

    hol_g     = 1.0    # Holstein coupling constant
    hol_omega = 1.0    # Phonon frequency

    tol        = 1e-6  # Tolerance for numerical computations
    nroots     = 1     # Number of roots to find (eigenvalues, for example)
    max_cycle  = 100   # Maximum number of iterations for a calculation
    max_space  = 500

    n_sweeps     = 10
    dav_max_iter = 30
    bond_dims    = [100] * 2 + [200] * 2 + [400] * 2
    noises       = [1e-4] * 2 + [1e-6] * 2 #  + [1e-5] * 4 + [0]
    thrds        = [1e-8] * 8
    
    def __init__(self, hol_g=1.0, hol_omega=1.0, nsite=4):
        """
        Constructor to initialize the Holstein model parameters.

        Parameters:
        - hol_g (float): Holstein coupling constant.
        - hol_omega (float): Phonon frequency.
        - nsite (int): Number of sites in the lattice.
        """
        self.hol_g = hol_g
        self.hol_omega = hol_omega
        self.nsite = nsite
        self.nmode = nsite  # Set nmode to be the same as nsite

        self.tmpdir   = os.environ.get('PYSCF_TMPDIR', lib.param.TMPDIR)
        self._chkfile = tempfile.NamedTemporaryFile(dir=self.tmpdir)
        self.chkfile  = self._chkfile.name

    def dump_info(self):
        log = lib.logger.new_logger(self, self.verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)

        d = self.__dict__.copy()
        for k in d.keys():
            log.info("%s = %s", k, str(d[k]))

        log.info("")

    def get_h1e(self):
        """
        Get the one-electron Hamiltonian for the tight-binding model.

        Returns:
        - ndarray: The one-electron Hamiltonian matrix.
        """
        return get_h1e_tb(self.nsite, self.nmode, self.is_pbc)

    def get_h1p1e(self):
        """
        Get the one-electron and one-phonon Hamiltonian with Holstein coupling.

        Returns:
        - ndarray: The one-electron and one-phonon Hamiltonian matrix.
        """
        return get_h1p1e_hol(self.nsite, self.nmode, self.hol_g)

    def get_h1p(self):
        """
        Get the one-phonon Hamiltonian for the Holstein model.

        Returns:
        - ndarray: The one-phonon Hamiltonian matrix.
        """
        return get_h1p_hol(self.nsite, self.nmode, self.hol_omega)

HolsteinModel = NoSpin1eHolsteinModel

if __name__ == "__main__":
    hol_omega = 5.0

    for alpha in [0.1, 1.0, 10.0, 100.0]: # 250.0]:
        hol_g = numpy.sqrt(hol_omega * alpha)
        hol_obj = HolsteinModel(hol_g=hol_g, hol_omega=hol_omega, nsite=4)
        hol_obj.is_pbc = True
        hol_obj.nph    = None
        hol_obj.verbose = 5
        hol_obj.nroots = 1
        hol_obj.tol = 1e-10

        print("\nalpha = %12.6f" % alpha)
        res = solve_hf(hol_obj, verbose=0)
