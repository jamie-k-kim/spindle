import numpy as np
from pyscf import tools, cc, fci, ao2mo
from pyscf.fci import selected_ci


# Runs PySCF classical preparation (HF, CCSD, FCI) from an FCIDUMP file.
# Returns all necessary integrals, reference energies, and T2 amplitudes.
def run_classical_prep(fcidump_path):
    # Load molecule & basic properties
    mf_as = tools.fcidump.to_scf(fcidump_path)
    norb = mf_as.mol.nao
    nela = mf_as.mol.nelectron // 2
    nelb = mf_as.mol.nelectron // 2
    nelec = (nela, nelb)

    # CCSD (To initialize LUCJ circuit)
    mycc = cc.CCSD(mf_as)
    mycc.kernel()
    t2_reference = mycc.t2

    # Exact reference (FCI or CCSD(T) fallback)
    if norb <= 12:
        try:
            cisolver = fci.FCI(mf_as)
            fci_energy, _ = cisolver.kernel()
        except Exception:
            fci_energy = mycc.e_tot + mycc.ccsd_t()
    else:
        fci_energy = mycc.e_tot + mycc.ccsd_t()

    # Extract integrals for SQD solver
    h0e = mf_as.mol.energy_nuc()
    h1e = mf_as.get_hcore()
    h2e = ao2mo.restore(1, mf_as._eri, norb)

    # Extract correlation weights (for chem-aware mapping)
    t2_corr = np.zeros((norb, norb))
    nocc = nela # assuming closed shell
    nvir = norb - nocc

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    val = abs(t2_reference[i, j, a, b])
                    t2_corr[i, nocc+a] += val
                    t2_corr[j, nocc+b] += val
                    t2_corr[i, nocc+b] += val
                    t2_corr[j, nocc+a] += val

    # Package and return
    return {
        "norb": norb,
        "nela": nela,
        "nelb": nelb,
        "nelec": nelec,
        "fci_energy": fci_energy,
        "h0e": h0e,
        "h1e": h1e,
        "h2e": h2e,
        "t2_reference": t2_reference,
        "t2_corr": t2_corr
    }


# Generates logical orbital interaction pairs based on the target topology.
def build_interaction_pairs(norb, topology, corr_matrix=None):

    alpha_beta = [(p, p) for p in range(norb)]
    
    if topology == "linear":
        alpha_alpha = [(p, p + 1) for p in range(norb - 1)]
        
    elif topology == "dense":
        alpha_alpha = [(p, q) for p in range(norb) for q in range(p + 1, norb)]
        
    elif topology == "chem_aware":
        if corr_matrix is None:
            raise ValueError("chem_aware topology requires a valid corr_matrix.")
            
        pairs = set([(p, p + 1) for p in range(norb - 1)])
        long_range_corrs = []
        for p in range(norb):
            for q in range(p + 2, norb): 
                long_range_corrs.append((corr_matrix[p, q], (p, q)))
        
        long_range_corrs.sort(key=lambda x: x[0], reverse=True)
        for _, pair in long_range_corrs[:3]:
            pairs.add(pair)
            
        alpha_alpha = list(pairs)
        
    else:
        raise ValueError(f"Unknown topology: {topology}")
        
    return (alpha_alpha, alpha_beta)


# Converts Qiskit bitstrings to PySCF CI integer strings, filtering out states that break particle number conservation.
def get_pyscf_ci_strings(bitstring, norb, nela, nelb):
    rev_str = bitstring[::-1] 
    alpha_bits = rev_str[:norb]
    beta_bits  = rev_str[norb:]
    if alpha_bits.count('1') == nela and beta_bits.count('1') == nelb:
        return int(alpha_bits[::-1], 2), int(beta_bits[::-1], 2)
    return None, None


def compute_sqd_error(counts, chem_data, shots, min_count=1):
    """
    Compute SQD energy and energy error from measurement counts.

    Parameters
    ----------
    counts : dict
        Measurement counts mapping bitstrings -> shot counts.
    chem_data : dict
        Chemistry problem data containing:
            h0e, h1e, h2e,
            norb, nelec,
            nela, nelb,
            fci_energy
    shots : int
        Total number of measurement shots.
    min_count : int
        Minimum count threshold to keep a determinant in the Selected CI subspace.

    Returns
    -------
    dict
        {
            "valid_ratio": float,
            "energy": float or np.nan,
            "energy_error": float or np.nan,
        }
    """

    valid_sa, valid_sb = set(), set()
    valid_shots = 0

    for bitstring, count in counts.items():
        a_int, b_int = get_pyscf_ci_strings(
            bitstring,
            chem_data["norb"],
            chem_data["nela"],
            chem_data["nelb"]
        )

        if a_int is not None:
            valid_shots += count
            if count >= min_count:
                valid_sa.add(a_int)
                valid_sb.add(b_int)

    valid_ratio = valid_shots / shots

    if len(valid_sa) == 0 or len(valid_sb) == 0:
        print(f"Fidelity (Valid Ratio): {valid_ratio:.2%}")
        print("No valid shots survived QPU noise. Diagonalization failed.")
        return {
            "valid_ratio": float(valid_ratio),
            "energy": float('nan'),
            "energy_error": float('nan'),
            "n_alpha_dets": 0,
            "n_beta_dets": 0,
        }

    try:
        sci_solver = selected_ci.SelectedCI()

        e_sqd, _ = selected_ci.kernel_fixed_space(
            sci_solver,
            chem_data["h1e"],
            chem_data["h2e"],
            chem_data["norb"],
            chem_data["nelec"],
            ci_strs=(
                sorted(valid_sa),
                sorted(valid_sb)
            ),
            verbose=0,
        )

        e_sqd_total = float(e_sqd + chem_data["h0e"])

        energy_error = float(abs(e_sqd_total - chem_data["fci_energy"]))

        print(f"Fidelity (Valid Ratio): {valid_ratio:.2%}")
        print(f"Energy Error: {energy_error:.15e} Ha")

        return {
            "valid_ratio": float(valid_ratio),
            "energy": e_sqd_total,
            "energy_error": energy_error,
            "n_alpha_dets": len(valid_sa),
            "n_beta_dets": len(valid_sb),
        }

    except Exception as exc:
        print(f"SQD diagonalization failed: {exc}")

        return {
            "valid_ratio": float(valid_ratio),
            "energy": float('nan'),
            "energy_error": float('nan'),
            "n_alpha_dets": len(valid_sa),
            "n_beta_dets": len(valid_sb),
        }