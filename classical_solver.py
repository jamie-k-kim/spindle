import numpy as np
from quantum_fragment_methods.application.solvers.classical_zoo.ccsd import CCSD
from quantum_fragment_methods.application.solvers.classical_zoo.fci import FCI
from quantum_fragment_methods.application.solvers.base import SolverResult

def run(mf, config, force_ccsd=False) -> SolverResult:
    """
    Run classical solver on a mean-field object.
    
    Args:
        mf: PySCF mean-field object (e.g. RHF).
        config: RunConfig dataclass.
        force_ccsd: If True, bypasses FCI and forces CCSD (e.g. for extracting T1/T2 amplitudes).
        
    Returns:
        SolverResult containing energy, t1, t2 amplitudes, and RDMs (if computed).
    """
    norb = mf.mol.nao
    if norb <= config.fci_threshold and not force_ccsd:
        print(f"     [Classical Solver] System size ({norb}) <= FCI Threshold ({config.fci_threshold}). Using exact FCI.")
        import pyscf.fci
        cisolver = pyscf.fci.FCI(mf)
        cisolver.verbose = 0
        energy, ci_vec = cisolver.kernel()
        dm1, dm2 = cisolver.make_rdm12(ci_vec, norb, mf.mol.nelec)
        
        result = SolverResult(
            energy=energy,
            wavefunction=ci_vec,
            rdm1=dm1,
            rdm2=dm2,
            metadata={"solver": "FCI"}
        )
    else:
        if not force_ccsd:
            print(f"     [Classical Solver] System size ({norb}) > FCI Threshold ({config.fci_threshold}). Using CCSD.")
        solver = CCSD()
        result = solver.solve(mf, compute_rdms=True)
    
    # Ensure result contains the necessary data
    return result

def run_from_integrals(h1e, h2e, norb, nocc, config, force_ccsd=False) -> SolverResult:
    """
    Run classical solver from given 1e and 2e integrals.
    
    Args:
        h1e: 1-electron integrals
        h2e: 2-electron integrals
        norb: Number of spatial orbitals
        nocc: Number of occupied spatial orbitals (alpha electrons)
        config: RunConfig dataclass
        force_ccsd: If True, bypasses FCI and forces CCSD (e.g. for extracting T1/T2 amplitudes).
        
    Returns:
        SolverResult
    """
    if nocc == 0 or nocc == norb:
        # Zero correlation energy case
        nvir = norb - nocc
        
        rdm1 = np.zeros((norb, norb))
        rdm2 = np.zeros((norb, norb, norb, norb))
        
        if nocc == norb:
            # Full occupancy
            np.fill_diagonal(rdm1, 2.0)
            for i in range(norb):
                for j in range(norb):
                    rdm2[i, i, j, j] += 4.0
                    rdm2[i, j, j, i] -= 2.0

        return SolverResult(
            energy=0.0,
            rdm1=rdm1,
            rdm2=rdm2,
            metadata={"t1": np.zeros((nocc, nvir)), "t2": np.zeros((nocc, nocc, nvir, nvir))}
        )

    if norb <= config.fci_threshold and not force_ccsd:
        print(f"     [Classical Solver] Fragment size ({norb}) <= FCI Threshold ({config.fci_threshold}). Using exact FCI.")
        solver = FCI()
    else:
        if not force_ccsd:
            print(f"     [Classical Solver] Fragment size ({norb}) > FCI Threshold ({config.fci_threshold}). Using CCSD.")
        solver = CCSD()
        
    return solver.solve_from_integrals(h1e, h2e, norb, nocc, compute_rdms=True)
