import sys, os, numpy
from itertools import combinations

import pyblock2
from pyblock2.driver import core # import DMRGDriver
from pyblock2.driver.core import SymmetryTypes
from pyblock2.driver.core import MPOAlgorithmTypes

import scipy, pyscf
from pyscf import lib
from pyscf.lib import logger

import utils

class Solver(lib.StreamObject):
    e0 = 0.0
    xi = None
    driver = None

    stat_list = None
    oper_list = None

    ferm_list = None
    bosn_list = None
    
    def add_term(self, opstr, opidx, value=1.0, expr_obj=None):
        log = lib.logger.new_logger(self, self.verbose)

        ferm_list = self.ferm_list
        assert len(opstr) == len(opidx)
        assert expr_obj is not None

        mask  = numpy.argsort(opidx)
        os = "".join([opstr[i] for i in mask])
        oi = [opidx[i] for i in mask]
        vv = value * (-1) ** (lambda a, b: sum(ai > aj and ai in b and aj in b for ai, aj in combinations(a, 2)))(opidx, ferm_list)

        # log.debug(f"Addterm: %4s, [%s], % 6.4f -> %4s, [%s], % 6.4f" % (opstr, ", ".join([str(i) for i in opidx]), value, os, ", ".join([str(i) for i in oi]), vv))
        expr_obj.add_term(os, oi, vv)

    def make_mpo(self, hf_obj, method=1):
        log = lib.logger.new_logger(self, self.verbose)
        log.info("\n\nBuilding the Hamiltonian in MPO form.")
        log.info("Method = %d" % method)

        if method == 1:
            const = hf_obj.const
            xi = hf_obj.xi
        else:
            const = 0.0
            xi = hf_obj.xi * 0.0

        driver = self.driver
        expr = driver.expr_builder()

        # electron part
        nsite = hf_obj.nsite
        nmode = hf_obj.nmode
        ferm_list = self.ferm_list
        bosn_list = self.bosn_list

        assert len(ferm_list) == nsite
        assert len(bosn_list) == nsite

        if method == 1:
            h1e = hf_obj.f1e   
            for kk, (ii, jj) in enumerate(zip(numpy.arange(nsite) % nsite, (numpy.arange(nsite) + 1)% nsite)):
                self.add_term("cd", [ferm_list[kk], ferm_list[kk]], h1e[kk, kk], expr_obj=expr)
                self.add_term("cd", [ferm_list[ii], ferm_list[jj]], h1e[ii, jj], expr_obj=expr)
                self.add_term("cd", [ferm_list[jj], ferm_list[ii]], h1e[jj, ii], expr_obj=expr)
                
        else:
            h1e = hf_obj.h1e
            for kk, (ii, jj) in enumerate(zip(numpy.arange(nsite) % nsite, (numpy.arange(nsite) + 1)% nsite)):
                self.add_term("cd", [ferm_list[ii], ferm_list[jj]], h1e[ii, jj], expr_obj=expr)
                self.add_term("cd", [ferm_list[jj], ferm_list[ii]], h1e[jj, ii], expr_obj=expr)

        # electron-phonon part
        g1p1e = hf_obj.g1p1e
        for (ii, kk), (jj, ll) in zip(enumerate(ferm_list), enumerate(bosn_list)):
            self.add_term("cdE", [kk, kk, ll], g1p1e[jj, ii, ii], expr_obj=expr)
            self.add_term("cdF", [kk, kk, ll], g1p1e[jj, ii, ii], expr_obj=expr)

        self.xi = hf_obj.xi
        hpp = hf_obj.hpp
        for ii, kk in enumerate(bosn_list):
            self.add_term("EF", [kk, kk], hpp[ii, ii], expr_obj=expr)

        if method == 1:
            for ii, kk in enumerate(bosn_list):
                self.add_term("E", [kk], -xi[ii] * hpp[ii, ii], expr_obj=expr)
                self.add_term("F", [kk], -xi[ii] * hpp[ii, ii], expr_obj=expr)

        mpo = driver.get_mpo(
            expr.finalize(adjust_order=False), 
            algo_type=MPOAlgorithmTypes.FastBipartite
            )
        
        log.info("Done building the Hamiltonian in MPO form.")
        
        return const, mpo
    
    def make_exp_na(self, mps=None):
        t0 = logger.process_clock(), logger.perf_counter()

        driver = self.driver
        expr = driver.expr_builder()

        ferm_list = self.ferm_list
        bosn_list = self.bosn_list
        nsite = len(ferm_list)
        nmode = len(bosn_list)

        assert len(ferm_list) == nsite
        assert len(bosn_list) == nsite

        exp_na_mo = numpy.zeros((nsite, nsite))
        for i, ii in enumerate(ferm_list):
            for j, jj in enumerate(ferm_list):
                t1 = logger.process_clock(), logger.perf_counter()
                expr = driver.expr_builder()
                self.add_term("cd", [ii, jj], 1.0, expr_obj=expr)

                mpo = driver.get_mpo(
                    expr.finalize(adjust_order=False), 
                    algo_type=MPOAlgorithmTypes.FastBipartite
                    )

                exp_na_mo[i, j] = driver.expectation(mps, mpo, mps, iprint=0)

                logger.timer(self, "exp_na %d %d" % (i, j), *t1)

        logger.timer(self, "exp_na", *t0)
        return exp_na_mo, exp_na_mo
    
    def make_exp_nb(self, mps=None):
        driver = self.driver
        expr = driver.expr_builder()

        xi = self.xi
        ferm_list = self.ferm_list
        bosn_list = self.bosn_list
        nsite = len(ferm_list)
        nmode = len(bosn_list)

        assert xi.shape == (nmode,)
        assert len(ferm_list) == nsite
        assert len(bosn_list) == nsite

        exp_nb = numpy.zeros((nmode, nmode))
        for i, ii in enumerate(bosn_list):
            for j, jj in enumerate(bosn_list):
                expr = driver.expr_builder()
                self.add_term("EF", [ii, jj], 1.0, expr_obj=expr)

                mpo = driver.get_mpo(
                    expr.finalize(adjust_order=False), 
                    algo_type=MPOAlgorithmTypes.FastBipartite
                    )

                exp_nb[i,j] = driver.expectation(mps, mpo, mps, iprint=0)

        exp_xb = numpy.zeros((nmode,))
        for i, ii in enumerate(bosn_list):
            expr = driver.expr_builder()
            self.add_term("E", [ii], 1.0, expr_obj=expr)
            self.add_term("F", [ii], 1.0, expr_obj=expr)

            mpo = driver.get_mpo(
                expr.finalize(adjust_order=False), 
                algo_type=MPOAlgorithmTypes.FastBipartite
                )

            exp_xb[i] = driver.expectation(mps, mpo, mps, iprint=0)

        exp_nb += numpy.einsum("I,J->IJ", xi-exp_xb, xi)

        return exp_nb
    
    def make_exp_xb(self, mps=None):
        log = lib.logger.new_logger(self, self.verbose)
        driver = self.driver
        expr = driver.expr_builder()

        ferm_list = self.ferm_list
        bosn_list = self.bosn_list
        nsite = len(ferm_list)
        nmode = len(bosn_list)

        assert len(ferm_list) == nsite
        assert len(bosn_list) == nsite

        exp_xb = numpy.zeros((nmode,))
        cpu0 = logger.process_clock(), logger.perf_counter()
        for i, ii in enumerate(bosn_list):
            expr = driver.expr_builder()
            self.add_term("E", [ii], 1.0, expr_obj=expr)
            self.add_term("F", [ii], 1.0, expr_obj=expr)

            mpo = driver.get_mpo(
                expr.finalize(adjust_order=False), 
                algo_type=MPOAlgorithmTypes.FastBipartite
                )

            exp_xb[i] = driver.expectation(mps, mpo, mps, iprint=0)
            t1 = log.timer("exp_xb %d" % i, *t1)

        return exp_xb
    
    def make_exp_na_xb(self, mps=None):
        driver = self.driver
        expr = driver.expr_builder()

        ferm_list = self.ferm_list
        bosn_list = self.bosn_list
        nsite = len(ferm_list)
        nmode = len(bosn_list)

        assert len(ferm_list) == nsite
        assert len(bosn_list) == nsite

        na_xb_mo = numpy.zeros((nsite, nsite, nmode))

        for i, ii in enumerate(bosn_list):
            for k, kk in enumerate(ferm_list):
                expr = driver.expr_builder()
                self.add_term("cdE", [kk, kk, ii], 1.0, expr_obj=expr)
                self.add_term("cdF", [kk, kk, ii], 1.0, expr_obj=expr)

                mpo = driver.get_mpo(
                    expr.finalize(adjust_order=False), 
                    algo_type=MPOAlgorithmTypes.FastBipartite
                    )

                na_xb_mo[k, k, i] = driver.expectation(mps, mpo, mps, iprint=0)

        return na_xb_mo

def build(hf_obj=None, verbose=0):
    """
    Build the Hamiltonian for the Holstein model. The method can be:
    - directep: will perform the variational polaron transformation and then 
                DMRG calculation.
    - directep-kernel: will assume a uniform distribution of the electron density
                    and perform the DMRG calculation.
    """

    nph  = hf_obj.nph + 1
    nsite_ferm = hf_obj.nsite
    nsite_bosn = hf_obj.nmode
    nsite_tot  = nsite_ferm + nsite_bosn

    nelec = hf_obj.nelec_alph + hf_obj.nelec_beta
    spin  = hf_obj.nelec_alph - hf_obj.nelec_beta

    driver = core.DMRGDriver(scratch=hf_obj.tmpdir, symm_type=SymmetryTypes.SZ, n_threads=20)
    driver.initialize_system(n_sites=nsite_tot, n_elec=nelec, spin=spin)

    # [[Part A]] Set states and matrix representation of operators in local Hilbert space
    stat_list = []
    oper_list = []

    ferm_list = []
    bosn_list = []
    
    # quantum number wrapper (n_elec, 2 * spin, point group irrep)
    quantum_number_wrapper = driver.bw.SX 

    for i in range(nsite_tot):
        if i % 2 == 0:
            stat = [ # [01]
                (quantum_number_wrapper(0,  0,  0), 1), 
                (quantum_number_wrapper(1,  1,  0), 1), 
                ] 
        
            oper = {
                "":  numpy.array([[1,  0], [0,  1]]), # identity
                "c": numpy.array([[0,  0], [1,  0]]), # alpha+
                "d": numpy.array([[0,  1], [0,  0]]), # alpha
            }

            stat_list.append(stat)
            oper_list.append(oper)
            ferm_list.append(i)

        else:
            stat = [(quantum_number_wrapper(0, 0, 0), nph)]
            oper = {
                "":  numpy.identity(nph), # identity
                "E": numpy.diag(numpy.sqrt(numpy.arange(1, nph)), k = -1),  # ph+
                "F": numpy.diag(numpy.sqrt(numpy.arange(1, nph)), k =  1),  # ph
            }

            stat_list.append(stat)
            oper_list.append(oper)
            bosn_list.append(i)

    assert nsite_bosn == nsite_ferm

    # [[Part B]] Set Hamiltonian terms in Holstein model
    driver.ghamil = driver.get_custom_hamiltonian(stat_list, oper_list)
    
    solver = Solver()
    solver.driver = driver
    solver.stat_list = stat_list
    solver.oper_list = oper_list
    solver.ferm_list = ferm_list
    solver.bosn_list = bosn_list
    return solver

def solve(hol_obj, **kwargs):
    log = lib.logger.new_logger(hol_obj, hol_obj.verbose)

    cur_tag = kwargs.get("cur_tag", None)
    pre_tag = kwargs.get("pre_tag", None)
    method  = getattr(hol_obj, "dmrg_method", 1)

    log.info("\n***********************************************")
    log.info("Running DMRG calculation for the Holstein model.")

    hol_obj.dump_info()
    nsite = hol_obj.nsite
    nmode = hol_obj.nmode

    cput0 = (logger.process_clock(), logger.perf_counter())

    cur_tag = cur_tag if cur_tag is not None else "hol-dmrg"
    if pre_tag is not None:
        pre_path = os.path.join(hol_obj.tmpdir, "%s-mps_info.bin" % pre_tag)
        if not os.path.isfile(pre_path):
            log.warn("Cannot find MPS file %s" % pre_path)
            pre_tag = None

    from utils import solve_hf
    hf_obj   = solve_hf(hol_obj, verbose=hol_obj.verbose)
    solver   = build(hf_obj, verbose=hol_obj.verbose)
    solver.verbose = hol_obj.verbose
    const, mpo = solver.make_mpo(hf_obj, method=method)

    if pre_tag is not None:
        log.info("Copy MPS from %s to %s" % (pre_tag, cur_tag))
        pre_mps = solver.driver.load_mps(pre_tag)
        mps = pre_mps.deep_copy(tag=cur_tag)
        mps.info.save_data(solver.driver.scratch + "/%s-mps_info.bin" % mps.info.tag)

    else:
        log.info("Generate random MPS")
        mps = solver.driver.get_random_mps(tag=cur_tag, bond_dim=hol_obj.bond_dims[0], nroots=1)

    res = solver.driver.dmrg(
        mpo, mps, n_sweeps=hol_obj.n_sweeps, 
        dav_max_iter=hol_obj.dav_max_iter,
        iprint=2, bond_dims=hol_obj.bond_dims,
        noises=hol_obj.noises, thrds=hol_obj.thrds,
        tol=hol_obj.tol
        )
    
    ee = res + const
    log.info("DMRG energy = % 12.8f" % ee)
    log.info("res = % 12.8f" % res)
    log.info("const = % 12.8f" % const)

    h1e_ao = hf_obj.h1e
    # na_mo, na_ao = solver.make_exp_na(mps=mps)

    hpp = hf_obj.hpp
    # nb = solver.make_exp_nb(mps=mps)

    hpx = hf_obj.hpx
    # xb = solver.make_exp_xb(mps=mps)

    h1e1p_ao = hf_obj.h1p1e.transpose(1, 2, 0)
    # na_xb_mo, na_xb_ao = solver.make_exp_na_xb(mps=mps)
    
    cput1 = logger.timer(hol_obj, "DMRG", *cput0)
    ll = "dmrg"
    data = {
        "ene_dmrg": res + const, "label": [ll],
        "ene_hf": hf_obj.ene,
        "h1e_ao": h1e_ao, "h1e1p_ao": h1e1p_ao, "hpp": hpp, "hpx": hpx,
        # "na_ao_%s" % ll : na_ao, "na_xb_ao_%s" % ll : na_xb_ao,
        # "nb_%s" % ll: nb, "xb_%s" % ll: xb, 
    }
    return data

if __name__ == "__main__":
    hol_omega = 4.0
    ene_list = []

    for hol_g in [4.0]: # numpy.linspace(0.0, 5.0, 21):    
        data = {}
        data["hol_g"] = hol_g

        for iph, nph in enumerate([10]):
            hol_obj = utils.HolsteinModel(hol_g=hol_g, hol_omega=hol_omega, nsite=4)
            hol_obj.is_pbc = True
            hol_obj.nph_max = None
            hol_obj.verbose = 0
            hol_obj.nroots = 1
            hol_obj.tol = 1e-10
            hol_obj.bond_dims = [20, 80, 80, 80, 160, 160]
        
            hol_obj.nph = nph
            res = solve(hol_obj, pre_tag=None)

            if iph == 0:
                data["ene_hf"] = res["ene_hf"]

            data["ene_fci_nph_%d" % nph] = res["ene_dmrg"]
        
        ene_list.append(data)
    
    first_line = "# " + (", ".join(["% 12s" % (key) for key in ene_list[0].keys()]))[2:]
    print(first_line)

    for data in ene_list:
        line = ",".join(["%12.8f" % (value) for key, value in data.items()])
        print(line)
