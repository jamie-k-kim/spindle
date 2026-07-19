import numpy as np

from quantum_fragment_methods.application.embedding.ewf import EWF
import classical_solver
import quantum_solver
from quantum_fragment_methods.application.solvers.base import SolverResult

def run(mf, config) -> float:
    """
    Run the EWF fragmentation workflow on a large molecule.
    
    Args:
        mf: PySCF mean-field object.
        config: RunConfig dataclass.
        
    Returns:
        total_energy: The reconstructed total EWF energy.
    """
    print(f"Full System Orbitals: {mf.mol.nao}")

    # EWF Fragmentation
    print(f"\n[Step 2] EWF Fragmentation (Bath: {config.ewf_bath_type}, Truncation: {config.ewf_truncation})...")
    embedder = EWF(bath_type=config.ewf_bath_type, truncation=config.ewf_truncation)
    embedding_result = embedder.kernel(mf, fragmentation="orbital")
    num_fragments = len(embedding_result.fragments)
    print(f"Molecule sliced into {num_fragments} fragments.")

    # Fragment Solutions
    print(f"\n[Step 3] Solving Fragments (Classical Threshold <= {config.classical_threshold} orbitals)...")
    fragment_results = {}
    
    num_quantum = 0
    num_classical = 0
    num_fallback = 0
    num_skipped = 0
    
    for frag_id, fragment in embedding_result.fragments.items():
        vfrag = fragment.metadata["vayesta_fragment"]
        norb = fragment.n_orbitals
        nelec = vfrag.cluster.nelec if hasattr(vfrag.cluster, 'nelec') else (fragment.n_electrons//2, fragment.n_electrons//2)
        n_alpha, n_beta = nelec
        
        # Get cluster integrals from vayesta fragment
        h1e, h2e = vfrag._hamil.get_integrals()
        
        nocc = min(norb, max(0, int(n_alpha)))
        print(f"  -> Fragment {frag_id}: {norb} orbitals, {n_alpha+n_beta} electrons")

        if nocc == 0 or nocc == norb:
            print(f"     Fragment {frag_id} has {nocc} electrons and {norb} orbitals. Skipping CCSD.")
            num_skipped += 1
            # For 0 electrons, no occupied orbitals. For full occupancy, no virtual orbitals.
            nvir = norb - nocc
            rdm1 = np.zeros((norb, norb))
            rdm2 = np.zeros((norb, norb, norb, norb))
            if nocc == norb:
                np.fill_diagonal(rdm1, 2.0)
                for i in range(norb):
                    for j in range(norb):
                        rdm2[i, i, j, j] += 4.0
                        rdm2[i, j, j, i] -= 2.0
            fragment_results[frag_id] = SolverResult(
                energy=0.0,
                rdm1=rdm1,
                rdm2=rdm2,
                metadata={"t1": np.zeros((nocc, nvir)), "t2": np.zeros((nocc, nocc, nvir, nvir))}
            )
        elif norb <= config.classical_threshold:
            print(f"     Routing to Classical CCSD Solver...")
            num_classical += 1
            result = classical_solver.run_from_integrals(h1e, h2e, norb, nocc, config)
            print(f"     Fragment {frag_id} Energy: {result.energy:.6f} Ha")
            fragment_results[frag_id] = result
        else:
            print(f"     Routing to Quantum SPIRAL Compiler...")
            # 1. Run local classical solver once to get t1 and t2 for this fragment
            c_res = classical_solver.run_from_integrals(h1e, h2e, norb, nocc, config, force_ccsd=True)
            t1 = c_res.metadata.get("t1")
            t2 = c_res.metadata.get("t2")
            
            # 2. Run quantum solver logic with the precomputed amplitudes
            q_res = quantum_solver.run(mf, config, t1=t1, t2=t2, norb=norb, nelec=(nocc, nocc), h1e=h1e, h2e=h2e, h0e=0.0)
            
            if q_res.rdm1 is None:
                print(f"     Fragment {frag_id} returned no RDMs from QPU (empty/failed). Falling back to classical.")
                num_fallback += 1
                q_res = c_res
                q_res.metadata["valid_ratio"] = "100.0% (Classical Fallback)"
            else:
                num_quantum += 1

            print(f"     Fragment {frag_id} SQD Energy: {q_res.energy:.6f} Ha")
            fragment_results[frag_id] = q_res

    # Energy Reconstruction
    print(f"\n[Step 4] Reconstructing Total Energy...")
    total_energy = embedder.reconstruct_energy(fragment_results, embedding_result)
    
    # Aggregate valid_ratio
    qpu_ratios = []
    for frag_id, res in fragment_results.items():
        vr = res.metadata.get("valid_ratio", "")
        if "%" in vr and "Classical" not in vr:
            try:
                val = float(vr.split("%")[0])
                qpu_ratios.append(val)
            except:
                pass
                
    if len(qpu_ratios) > 0:
        overall_ratio = f"{sum(qpu_ratios)/len(qpu_ratios):.2f}%"
    else:
        overall_ratio = "100.0% (Exact Classical)"
        
    return SolverResult(
        energy=total_energy,
        metadata={
            "valid_ratio": overall_ratio,
            "total_fragments": num_fragments,
            "quantum_fragments": num_quantum,
            "classical_fragments": num_classical,
            "fallback_fragments": num_fallback,
            "skipped_fragments": num_skipped
        }
    )
