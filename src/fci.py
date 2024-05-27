import sys, os, numpy, scipy, pyscf

from pyscf import lib
from pyscf.lib import logger

from utils import HolsteinModel, solve_hf

def build_fci_obj(hf_obj, n=None, verbose=0):
    coeff_e = hf_obj.coeff_e
    ene_hf  = hf_obj.ene

    h1e   = hf_obj.h1e
    f1e   = hf_obj.f1e
    h1p1e = hf_obj.g1p1e
    hpp   = hf_obj.hpp
    hpx   = hf_obj.hpx
    xi    = hf_obj.xi
    const = hf_obj.const

    class ElectronPhononFullConfigurationInteractionProblem(object):
        pass

    fci_obj = ElectronPhononFullConfigurationInteractionProblem()
    fci_obj.f1e   = numpy.einsum("mn,mp,nq->pq",   f1e,   coeff_e, coeff_e)
    fci_obj.g1p1e = numpy.einsum("Imn,mp,nq->Ipq", h1p1e, coeff_e, coeff_e)
    fci_obj.hpp = hpp
    fci_obj.hpx = hpx
    fci_obj.xi  = xi
    fci_obj.e0 = ene_hf
    fci_obj.const = const # * 0.0 

    return fci_obj

def solve(hol_obj, **kwargs):
    log = lib.logger.new_logger(hol_obj, hol_obj.verbose)

    cur_tag = kwargs.get("cur_tag", None)
    pre_tag = kwargs.get("pre_tag", None)

    log.info("\n***********************************************")
    log.info("Running FCI calculation for the Holstein model.")

    hol_obj.dump_info()
    cput0 = (logger.process_clock(), logger.perf_counter())

    if pre_tag is not None:
        path = os.path.join(hol_obj.tmpdir, pre_tag + ".h5")
        log.info("\nReading initial guess from %s", path)

        data = lib.chkfile.load(path, "fci")

    hf_obj  = solve_hf(hol_obj,     verbose=hol_obj.verbose)
    fci_obj = build_fci_obj(hf_obj, verbose=hol_obj.verbose)

    import epcc
    from epcc.fci import make_shape
    from epcc.fci import make_hdiag
    from epcc.fci import contract_1e
    from epcc.fci import contract_ep_rspace
    from epcc.fci import contract_pp
    from epcc.fci import make_rdm1e

    g1e1p = fci_obj.g1p1e.transpose(1, 2, 0)

    shape = make_shape(
        hol_obj.nsite, hol_obj.nelec_alph + hol_obj.nelec_beta,
        hol_obj.nmode, hol_obj.nph, e_only=False
    )
    size = numpy.prod(shape)

    hdiag = make_hdiag(
        fci_obj.f1e, 0.0, g1e1p, numpy.diag(fci_obj.hpp),
        hol_obj.nsite, hol_obj.nelec_alph + hol_obj.nelec_beta,
        hol_obj.nmode, hol_obj.nph, e_only=False, space="r"
    )

    assert hdiag.shape == (size,)

    def hop(v):
        assert v.shape == (size,)
        hv = contract_1e(fci_obj.f1e, v, hol_obj.nsite, hol_obj.nelec_alph + hol_obj.nelec_beta, hol_obj.nmode, hol_obj.nph, e_only=False, space="r")
        hv += contract_ep_rspace(g1e1p, v, hol_obj.nsite, hol_obj.nelec_alph + hol_obj.nelec_beta, hol_obj.nmode, hol_obj.nph)
        hv += contract_pp(numpy.diag(fci_obj.hpp), v, hol_obj.nsite, hol_obj.nelec_alph + hol_obj.nelec_beta, hol_obj.nmode, hol_obj.nph, xi=fci_obj.xi)
        return hv
    
    v0 = numpy.zeros(size)
    hdiag0 = hdiag.min()
    mask = (hdiag < hdiag0 + hol_obj.tol)
    v0[mask] = 1.0
    v0 += numpy.random.rand(size) * hol_obj.tol

    precond = lambda x, e, *args: x / (hdiag - e + hol_obj.tol)

    res = lib.davidson(
        hop, v0, precond, tol=hol_obj.tol, max_cycle=hol_obj.max_cycle,
        max_space=hol_obj.max_space, verbose=log
    )

    c = res[1].reshape(shape)
    data = {"ene_fci": res[0] + fci_obj.const, "ene_hf": hf_obj.ene, "c": c}

    rdm1e = make_rdm1e(c, hol_obj.nsite, hol_obj.nelec_alph + hol_obj.nelec_beta)
    rdm_site = numpy.einsum("mi,nj,ij->mn", hf_obj.coeff_e, hf_obj.coeff_e, rdm1e)
    print(rdm_site)

    if cur_tag is not None:
        path = os.path.join(hol_obj.tmpdir, cur_tag + ".h5")
        log.info("\nWriting results to %s", path)
        lib.chkfile.dump(path, "fci", data)

    cput1 = logger.timer(hol_obj, 'FCI', *cput0)
    return data

if __name__ == "__main__":
    nsite = 4
    hol_omega = 4.0
    ene_list = []

    for hol_g in [4.0]: # numpy.linspace(0.0, 10.0, 21):
        data = {}
        data["hol_g"] = hol_g

        for inph, nph in enumerate([10, 20, 40]):
            hol_obj = HolsteinModel(hol_g=hol_g, hol_omega=hol_omega, nsite=nsite)
            hol_obj.is_pbc = True
            hol_obj.nph_max = None
            hol_obj.verbose = 0
            hol_obj.nroots = 1
            hol_obj.tol = 1e-10
        
            hol_obj.nph = nph
            res = solve(hol_obj)

            if inph == 0:
                data["ene_hf"] = res["ene_hf"]

            # data["fci_nph_%d" % nph] = res["ene_fci"]

            print("\nhol_omega = % 6.4f, hol_g = % 6.4f, nph = %d" % (hol_omega, hol_g, nph))
            print(res["c"].shape)
            cc = res["c"].reshape(nsite, -1)
            cc2 = cc ** 2
            assert abs(cc2.sum() - 1.0) < 1e-10

            print("")
            print("- c[0] = %6.4f / %6.4f" % (res["c"][0, 0, 0, 0, 0, 0] ** 2, (res["c"][0, 0] ** 2).sum()))
            # numpy.savetxt(sys.stdout, res["c"][0, 0], fmt="% 6.4f", delimiter=", ")
            print("- c[1] = %6.4f / %6.4f" % (res["c"][1, 0, 0, 0, 0, 0] ** 2, (res["c"][1, 0] ** 2).sum()))
            # numpy.savetxt(sys.stdout, res["c"][1, 0], fmt="% 6.4f", delimiter=", ")
            print("\n")

            # data["c1_%d" % nph] = (res["c"][0, 0] ** 2).sum()
            # data["c2_%d" % nph] = (res["c"][1, 0] ** 2).sum()
            data["e_%d" % nph]  = res["ene_fci"]
        
        ene_list.append(data)

        
    max_len = max([len(key) for key, value in ene_list[0].items()])
    slen = max(12, max_len + 2)
    
    first_line = "# " + ", ".join(["%*s" % (slen, key) for key in ene_list[0].keys()])[2:]
    print(first_line)

    for data in ene_list:
        line = ", ".join([f"% {slen}.4f" % (value) for key, value in data.items()])
        print(line)
