import sys, os, inspect, tempfile
from functools import reduce
from typing import Any

import numpy, scipy, opt_einsum
from opt_einsum import contract as einsum

import pyscf
from pyscf import lib
from pyscf.lib import logger

from utils import HolsteinModel, solve_hf

def build_cc_obj(hf_obj, method=None, verbose=0):
    nn = None

    nsite = hf_obj.nsite
    nmode = hf_obj.nmode
    norb = nsite
    nocc = hf_obj.nelec_alph + hf_obj.nelec_beta
    nvir = norb - nocc

    if method == 1:
        nn = numpy.zeros(nsite)
        nn[0] = 1.0
        xi_rhs = einsum("Imm,m->I", hf_obj.h1p1e, nn)
        xi     = numpy.linalg.solve(hf_obj.hpp, xi_rhs)
    elif method == 2:
        nn = numpy.ones(nsite) / nsite
        xi_rhs = einsum("Imm,m->I", hf_obj.h1p1e, nn)
        xi     = numpy.linalg.solve(hf_obj.hpp, xi_rhs)
    else:
        xi = hf_obj.xi

    f1e = hf_obj.h1e - einsum("Imn,I->mn", hf_obj.h1p1e, xi, optimize=True) * 2.0
    ene_mo, coeff_e = scipy.linalg.eigh(f1e)
    occidx, viridx = (lambda m: (m[:nocc], m[nocc:]))(numpy.argsort(ene_mo))
    orbo = coeff_e[:, occidx].reshape(nsite, nocc)
    orbv = coeff_e[:, viridx].reshape(nsite, nsite - nocc)

    rho = einsum("mi,ni->mn", orbo, orbo)
    xi_rhs = einsum("Imn,mn->I", hf_obj.h1p1e, rho)
    xi     = numpy.linalg.solve(hf_obj.hpp, xi_rhs)

    ene_hf  = einsum("mn,mn->", hf_obj.h1e, rho)
    ene_hf -= einsum("IJ,I,J->", hf_obj.hpp, xi, xi)

    h1p1e = hf_obj.h1p1e
    hpp   = hf_obj.hpp
    hpx   = einsum("IJ,J->I", hpp, -xi)
    hpx  += einsum("Imn,mn->I", h1p1e, rho)

    class H1eBlocks(object):
        pass

    class H1p1eBlocks(object):
        pass

    class ElectronPhononCoupledClusterProblem(object):
        pass

    class Delta(object):
        pass

    h1e_blocks = H1eBlocks()
    h1e_blocks.vv = einsum("mn,mp,nq->pq", f1e, orbv, orbv)
    h1e_blocks.oo = einsum("mn,mp,nq->pq", f1e, orbo, orbo)
    h1e_blocks.vo = einsum("mn,mp,nq->pq", f1e, orbv, orbo)
    h1e_blocks.ov = einsum("mn,mp,nq->pq", f1e, orbo, orbv)

    h1p1e_blocks = H1p1eBlocks()
    h1p1e_blocks.vv = einsum("Imn,mp,nq->Ipq", h1p1e, orbv, orbv)
    h1p1e_blocks.oo = einsum("Imn,mp,nq->Ipq", h1p1e, orbo, orbo)
    h1p1e_blocks.vo = einsum("Imn,mp,nq->Ipq", h1p1e, orbv, orbo)
    h1p1e_blocks.ov = einsum("Imn,mp,nq->Ipq", h1p1e, orbo, orbv)

    cc_obj = ElectronPhononCoupledClusterProblem()
    cc_obj.coeff_e = coeff_e
    cc_obj.e0    = ene_hf
    cc_obj.h1e   = h1e_blocks
    cc_obj.h1p1e = h1p1e_blocks
    cc_obj.hpp   = hpp
    cc_obj.hpx   = hpx
    cc_obj.xi    = xi
    cc_obj.rho   = rho

    cc_obj.norb  = nsite
    cc_obj.nocc  = nocc
    cc_obj.nvir  = nsite - nocc
    cc_obj.nmode = nmode
    cc_obj.nsite = nsite

    cc_obj.delta = Delta()
    cc_obj.delta.oo = numpy.eye(nocc)
    cc_obj.delta.vv = numpy.eye(nvir)

    return cc_obj

class Solver(lib.StreamObject):
    """
    Class representing the amplitude for various quantum chemical methods.

    Attributes:
        - None defined in this basic structure.

    Methods:
        - init_guess: Initialize the amplitude guess based on a given object.
        - get_shape: Get the shape of the amplitude, needs to be implemented.
        - get_ene: Get the energy based on amplitude, needs to be implemented.
        - get_res: Get the residuals based on amplitude, needs to be implemented.
        - amplitude_to_vector: Convert a multi-dimensional amplitude array to a 1D vector.
        - vector_to_amplitude: Convert a 1D vector back to the multi-dimensional amplitude array.
    """
    max_cycle = 50
    conv_tol  = 1e-8
    max_trial = 5

    def __init__(self, m=None):
        self.m = m
        pass

    def init_guess(self, cc_obj=None):
        """
        Initialize the amplitude guess based on a given coupled-cluster object or similar.

        Parameters:
            - cc_obj: Coupled-cluster or similar object to initialize the amplitude guess.
        """
        pass

    def get_shape(self, cc_obj=None):
        """
        Get the shape of the amplitude. Specifics need to be implemented.

        Parameters:
            - cc_obj: Coupled-cluster or similar object.
        """
        raise NotImplementedError
    
    def get_ene(self, amp=None, cc_obj=None):
        """
        Get the energy based on the amplitude. Specifics need to be implemented.

        Parameters:
            - amp: The amplitude array.
            - cc_obj: Coupled-cluster or similar object.
        """
        raise NotImplementedError

    def get_res(self, amp=None, cc_obj=None):
        """
        Get the residuals based on amplitude. Specifics need to be implemented.

        Parameters:
            - amp: The amplitude array.
            - cc_obj: Coupled-cluster or similar object.
        """
        raise NotImplementedError
    
    def amplitude_to_vector(self, amp, shape=None):
        """
        Convert a multi-dimensional amplitude array to a 1D vector.

        Parameters:
            - amp: The amplitude array.
            - shape: The expected shape of the amplitude.

        Returns:
            - A 1D numpy array representing the amplitude.
        """
        vec = []
        for (ia, a), s in zip(enumerate(amp), shape):
            assert a.shape == s
            vec.append(a.flatten())
        return numpy.concatenate(vec)

    def vector_to_amplitude(self, vec, shape=None):
        """
        Convert a 1D vector back to the multi-dimensional amplitude array.

        Parameters:
            - vec: The 1D vector representing the amplitude.
            - shape: The expected shape of the amplitude.

        Returns:
            - A multi-dimensional amplitude array.
        """
        amp = []

        start = 0
        end = 0

        for ia, s in enumerate(shape):
            size = numpy.prod(s)
            end += size

            a = vec[start:end].reshape(s)
            amp.append(a)

            start = end

        return amp
    
    def pack_amplitude(self, amp, shape=None):
        raise NotImplementedError
    
    def unpack_amplitude(self, amp, shape=None):
        raise NotImplementedError
    
    def dump_info(self, verbose=None):
        log = lib.logger.new_logger(self, verbose)

        log.info('\n')
        log.info('******** %s ********', self.__class__)

        d = self.__dict__.copy()            
        log.info('Solver attributes:')
        for k, v in d.items():
            if k.startswith('_'):
                continue

            if isinstance(v, numpy.ndarray):
                log.info(f'{k:10s} = {v.shape}')
            elif isinstance(v, float):
                log.info(f'{k:10s} = {v: 12.8f}')
            else:
                log.info(f'{k:10s} = {v}')

    def kernel(self, amp0=None, vec0=None, cc_obj=None, verbose=None, max_cycle=None, conv_tol=None):
        log = lib.logger.new_logger(self, verbose)
        self.dump_info(verbose=verbose)

        assert cc_obj is not None
        shape = self.get_shape(cc_obj=cc_obj)
        size  = numpy.sum([numpy.prod(s) for s in shape])

        log.info('\n')
        if amp0 is not None:
            log.info('Using the given amplitude as the initial guess')
            assert vec0 is None, "amp0 and vec0 shall not be given at the same time"

            if isinstance(amp0, list):
                vec0 = self.amplitude_to_vector(amp0, shape=shape)

            elif isinstance(amp0, dict): # TODO: implement pack_amplitude and unpack_amplitude
                amp0 = self.unpack_amplitude(amp0, shape=shape)
                vec0 = self.amplitude_to_vector(amp0, shape=shape)

            else:
                raise TypeError("amp0 shall be a list or a dict")

        if vec0 is None:
            amp0 = self.init_guess(cc_obj=cc_obj)
            vec0 = self.amplitude_to_vector(amp0, shape=shape)

        assert amp0 is not None
        assert vec0 is not None
        assert vec0.size == size
        
        max_cycle = max_cycle if max_cycle is not None else self.max_cycle
        conv_tol  = conv_tol  if conv_tol  is not None else self.conv_tol

        # Calculate the initial residuals
        res0 = self.amplitude_to_vector(self.get_res(amp0, cc_obj=cc_obj), shape=shape)
        # Calculate the initial total energy
        ene0 = cc_obj.e0 + self.get_ene(amp0, cc_obj=cc_obj)

        # Log the initial energy values and residual norm
        log.info('Mean-field energy          = % 12.8f', cc_obj.e0)
        log.info('Initial correlation energy = % 12.8f', ene0 - cc_obj.e0)
        log.info('Initial total energy       = % 12.8f', ene0)
        log.info('Initial residual norm      = % 12.4e', numpy.linalg.norm(res0))

        if max_cycle > 0:
            # Import the optimize module from scipy
            cput0 = (logger.process_clock(), logger.perf_counter())
            from scipy import optimize
            from scipy.optimize._nonlin import NoConvergence

            vec_list = []
            ene_list = []
            err_list = []

            for itrial in range(self.max_trial):
                global iter_cc
                iter_cc = 0

                def func(vec):
                    amp = self.vector_to_amplitude(vec, shape=shape)

                    with opt_einsum.shared_intermediates():
                        res = self.get_res(amp=amp, cc_obj=cc_obj)
                        ene = self.get_ene(amp=amp, cc_obj=cc_obj)

                    res = self.amplitude_to_vector(res, shape=shape)

                    global iter_cc
                    iter_cc += 1
                    

                    log.info('iter %4d, energy = %12.8f, residual = %12.4e',
                            iter_cc, ene, numpy.linalg.norm(res))

                    assert iter_cc <= max_cycle, vec
                    return res

                v0 = None
                if itrial == 0:
                    v0 = vec0

                elif itrial == 1:
                    v0 = vec0 * 0.0

                else:
                    v0 = numpy.random.rand(vec0.size)

                vec_sol = None
                try:
                    # Solve the amplitude equations using the Newton-Krylov method
                    log.info('\nSolving amplitude equations using Newton-Krylov method')
                    options = {
                        'x_tol': numpy.sqrt(conv_tol),
                        'f_tol': numpy.sqrt(conv_tol),
                        'maxiter': max_cycle,
                        'verbose': (log.verbose >= 5),
                    }

                    for k, v in options.items():
                        if isinstance(v, float):
                            log.info('%s = % 6.4e', k, v)
                        else:
                            log.info('%s = %s', k, v)

                    vec_sol = optimize.newton_krylov(
                        func, v0, **options
                    )

                except Exception as e:
                    # If the method does not converge, log a warning and use the last solution
                    log.warn('Newton-Krylov method did not converge')
                    vec_sol = e.args[0]

                    if not isinstance(vec_sol, numpy.ndarray):
                        raise RuntimeError('vec_sol is not a numpy array')

                except AssertionError as e:
                    log.warn('Maximum number of iterations reached')
                    vec_sol = e

                assert vec_sol is not None

                amp_sol  = self.vector_to_amplitude(vec_sol, shape=shape)
                ene_sol  = cc_obj.e0 + self.get_ene(amp_sol, cc_obj=cc_obj)
                err_sol  = numpy.linalg.norm(self.amplitude_to_vector(self.get_res(amp_sol, cc_obj=cc_obj), shape=shape))
                err_sol /= size

                vec_list.append(vec_sol)
                ene_list.append(ene_sol)
                err_list.append(err_sol)

            # Find the solution with the lowest residual norm
            if self.max_trial > 1:
                log.info('\nSummary:')
                vec_list = numpy.array(vec_list)
                ene_list = numpy.array(ene_list)
                err_list = numpy.array(err_list)

                mask = numpy.array(err_list) < conv_tol
                if numpy.any(mask): #  'No solution found'
                    # out of mask, find ires that has the lowest energy
                    ires = numpy.nonzero(mask)[0][numpy.argmin(ene_list[mask])]
                else:
                    ires = numpy.argmin(err_list)
                    log.warn('No solution found')

                log.info('Selected solution:')
                for i, (v, e, r) in enumerate(zip(vec_list, ene_list, err_list)):
                    if i == ires:
                        info = "trial = %4d, energy = % 12.8f, residual = % 6.4e (selected)" % (i, e, r)
                        log.info(info)
                    else:
                        info = "trial = %4d, energy = % 12.8f, residual = % 6.4e" % (i, e, r)
                        log.info(info)
            else:
                ires = 0

            vec_sol = vec_list[ires]
            amp_sol = self.vector_to_amplitude(vec_sol, shape=shape)
            ene_sol = ene_list[ires]
            err_sol = err_list[ires]

            # Log the final energy values and residual norm
            log.info('\nFinal correlation energy    = % 12.8f', ene_sol - cc_obj.e0)
            log.info('Final total energy          = % 12.8f', ene_sol)
            log.info('Final residual norm         = % 12.4e, conv_tol = %12.4e\n', err_sol, conv_tol)
            self.is_converged = err_sol < conv_tol

        else:
            log.info('max_cycle = %d, skip the amplitude equations' % self.max_cycle)
            ene_sol = ene0
            res_sol = res0
            err_sol = numpy.linalg.norm(res0) / size
            amp_sol = amp0
            self.is_converged = True

        # Return the final total energy, correlation energy, and amplitudes
        return ene_sol, ene_sol - cc_obj.e0, self.pack_amplitude(amp_sol, shape=shape), err_sol
    
def build(e_order=1, p_order=2, path="./lib/"):
    import os, importlib.util

    file_path = os.path.join(path, f"epcc_e{e_order}_p{p_order}.py")
    assert os.path.exists(file_path)

    spec   = importlib.util.spec_from_file_location('amp', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.einsum = einsum

    res_list = []
    for ie in range(e_order + 1):
        for ip in range(p_order + 1):
            ires = 2 * ip - 1 + ie
            if ires < 0:
                continue

            res_info = {}
            res_info["ires"] = ires
            res_info["ie"] = ie
            res_info["ip"] = ip
            res_info["func"] = getattr(module, f"res_{ires}")

            res_list.append(res_info)

    res_list = sorted(res_list, key=lambda res_info: res_info["ires"])

    class GeneratedSolver(Solver):
        e_order = None
        p_order = None

        def init_guess(self, cc_obj=None):
            f1e = cc_obj.h1e
            hpp = cc_obj.hpp
            h1p1e = cc_obj.h1p1e
            hpx = cc_obj.hpx

            eo = f1e.oo.diagonal()
            ev = f1e.vv.diagonal()
            ep = hpp.diagonal()
            no, nv = f1e.ov.shape
            np = hpp.shape[0]

            amp0 = []
            for ires, res_info in enumerate(res_list):
                ie = res_info["ie"]
                ip = res_info["ip"]
                assert ires == res_info["ires"]

                is_t1  = (ie == 1 and ip == 0)
                is_s1  = (ie == 0 and ip == 1)
                is_u11 = (ie == 1 and ip == 1)
                shape = tuple([np] * ip + [nv, no] * ie)
 
                amp = numpy.zeros(shape)
                gap = None

                if is_t1:
                    gap = (ev[:, None] - eo[None, :])
                    amp = - f1e.vo
                
                elif is_s1:
                    gap = ep
                    amp = -hpx
                
                elif is_u11:
                    gap = (ep[:,None,None] + ev[None,:,None] - eo[None,None,:])
                    amp = -h1p1e.vo

                if gap is not None:
                    d = numpy.where(numpy.abs(gap) < self.conv_tol, 1.0, gap)
                    amp /= d

                assert amp.shape == shape
                amp0.append(amp.reshape(shape))

            return amp0

        def get_shape(self, cc_obj=None):
            f1e = cc_obj.h1e
            hpp = cc_obj.hpp
            h1p1e = cc_obj.h1p1e
            hpx = cc_obj.hpx

            no, nv = f1e.ov.shape
            np = hpp.shape[0]

            shape = []
            for ires, res_info in enumerate(res_list):
                ie = res_info["ie"]
                ip = res_info["ip"]
                assert ires == res_info["ires"]

                shape.append((np,) * ip + (nv, no,) * ie)

            return shape
        
        def get_ene(self, amp=None, cc_obj=None):
            ene = getattr(module, "ene")(cc_obj, amp)
            return ene
        
        def get_res(self, amp=None, cc_obj=None):
            res = []

            for ires, res_info in enumerate(res_list):
                assert ires == res_info["ires"]
                res.append(res_info["func"](cc_obj, amp))

            return res
        
        def pack_amplitude(self, amp, shape=None):
            assert amp is not None

            amp_dict = {}
            amp_dict["e_order"] = self.e_order
            amp_dict["p_order"] = self.p_order

            for (ires, res_info), s in zip(enumerate(res_list), shape):
                assert ires == res_info["ires"]
                ie = res_info["ie"]
                ip = res_info["ip"]

                assert amp[ires].shape == s
                amp_dict[f"{ie}-{ip}"] = amp[ires]

            return amp_dict
        
        def unpack_amplitude(self, amp_dict, shape=None):
            log = lib.logger.new_logger(self, self.verbose)

            assert amp_dict is not None

            if amp_dict["e_order"] != self.e_order or amp_dict["p_order"] != self.p_order:
                log.info("Transforming the amplitude from e_order = %d, p_order = %d to e_order = %d, p_order = %d" % (amp_dict["e_order"], amp_dict["p_order"], self.e_order, self.p_order))

            amp = []
            for (ires, res_info), s in zip(enumerate(res_list), shape):
                assert ires == res_info["ires"]
                ie = res_info["ie"]
                ip = res_info["ip"]
                
                a = amp_dict.get(f"{ie}-{ip}", None)

                if a is not None:
                    assert a.shape == s
                    amp.append(a)
                else:
                    log.info("The amplitude %d-%d is not found in the given amplitude dict, set it to zero." % (ie, ip))
                    amp.append(numpy.zeros(s))

            return amp

    amp_obj = GeneratedSolver()
    amp_obj.e_order = e_order
    amp_obj.p_order = p_order

    return amp_obj

def solve(hol_obj, **kwargs):
    log = lib.logger.new_logger(hol_obj, hol_obj.verbose)

    nph      = hol_obj.nph
    method   = kwargs.get("method", "with_oo")
    assert method == "with_oo"

    cur_tag  = kwargs.get("cur_tag", None)
    pre_tag  = kwargs.get("pre_tag", None)
    lib_path = kwargs.get("lib_path", "./lib/")

    log.info("\n***********************************************")
    log.info("Running EP-CC calculation for the Holstein model.")

    hol_obj.dump_info()
    cput0 = (logger.process_clock(), logger.perf_counter())

    hf_obj = solve_hf(hol_obj,     verbose=hol_obj.verbose)

    tmpdir = hf_obj.tmpdir
    tmp = os.path.join(tmpdir, "tmp.h5")
    from pyscf.lib.chkfile import save
    save(tmp, "hol", hf_obj.__dict__)
    save(tmp, "hol_g", hol_obj.hol_g)
    save(tmp, "hol_omega", hol_obj.hol_omega)

    nsite = hol_obj.nsite
    nelec = hol_obj.nelec_alph + hol_obj.nelec_beta
    nmode = hol_obj.nmode

    solver = build(1, nph, path=lib_path)
    solver.m = hol_obj
    solver.verbose = hol_obj.verbose
    solver.max_cycle = 100
    solver.conv_tol  = 1e-4

    res_list = []
    for method in [0, 1, 2]:
        log.info("\n***********************************************")
        log.info("Running EP-CC calculation for the Holstein model with method = %d." % method)
        cc_obj = build_cc_obj(hf_obj, method=method, verbose=hol_obj.verbose)
        res    = solver.kernel(cc_obj=cc_obj, amp0=None)
        cc_ene, hf_ene, amp, err = res

        res_list.append({
            "cc_obj": cc_obj, "err": err, 
            "cc_ene": cc_ene if solver.is_converged else numpy.nan,
            "cc_ene_": cc_ene, "amp": amp, "err": err,
            "hf_ene": hf_ene, "is_converged": solver.is_converged,
            "rho": cc_obj.rho, "xi": cc_obj.xi
        })
    
    log.info("\n***********************************************")
    log.info("Summary of the EP-CC calculations for the Holstein model.")
    
    # Select the result with error < tol
    err_list = numpy.array([res["err"]    for res in res_list])
    ene_list = numpy.array([res["cc_ene"] for res in res_list])
    mask = numpy.array(err_list) < solver.conv_tol
    
    if numpy.any(mask):
        ires = numpy.nonzero(mask)[0][numpy.argmin(ene_list[mask])]
    else:
        log.warn("No solution found")
        ires = numpy.argmin(ene_list)
    
    for i, res in enumerate(res_list):
        info = "method = %d, cc_ene = % 12.8f, err = % 6.4e" % (i, res["cc_ene"], res["err"])

        if i == ires:
            info += " (selected)"

        if not res["is_converged"]:
            info += " (not converged)"
            
        log.info(info)

    cc_obj = res_list[ires]["cc_obj"]
    solver.max_cycle = hol_obj.max_cycle
    solver.conv_tol  = hol_obj.tol
    solver.max_trial = 1
    res = solver.kernel(cc_obj=cc_obj, amp0=res_list[ires]["amp"])

    data = {
        "ene_mf": hf_obj.ene, "rho_mf": hf_obj.rho, "xi_mf": hf_obj.xi,
        
        "ene_cc": res[0],
        "ene_ref": res_list[ires]["hf_ene"],
        "rho_ref": cc_obj.rho, "xi_ref": cc_obj.xi,
    }
    
    return data

if __name__ == "__main__":
    hol_omega = 4.0
    ene_list = []

    for hol_g in [4.0]: # numpy.linspace(0.0, 5.0, 21):
        data = {}
        data["hol_g"] = hol_g

        amp = None

        for nph in [2]:
            hol_obj = HolsteinModel(hol_g=hol_g, hol_omega=hol_omega, nsite=4)
            hol_obj.is_pbc = True
            hol_obj.verbose = 5
            hol_obj.nroots = 1
            hol_obj.tol = 1e-5
            hol_obj.max_cycle = 100
        
            hol_obj.nph = nph
            res = solve(hol_obj, amp0=None)

            if nph == 1:
                data["ene_hf"] = res["ene_hf"]

            data["ene_cc_%d" % nph] = res["ene_cc"]
        
        ene_list.append(data)

    first_line = "# " + (", ".join(["% 12s" % (key) for key in ene_list[0].keys()]))[2:]
    print(first_line)

    for data in ene_list:
        line = ", ".join(["% 12.8f" % (value) for key, value in data.items()])
        print(line)
