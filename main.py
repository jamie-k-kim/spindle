import os
import time
import datetime
import argparse
import numpy as np

# PySCF Imports
from pyscf import tools
import pyscf.gto

# Local Imports
import classical_solver
import quantum_solver
import frag_manager

def patch_abstract_orbitals(mf):
    """
    Patches PySCF and Vayesta methods to support abstract orbital spaces
    without geometry (e.g. from FCIDUMP files).
    """
    original_get_veff = mf.get_veff.__func__
    def patched_get_veff(self, *args, **kwargs):
        if getattr(self, '_eri', None) is None:
            self._eri = mf._eri
        return original_get_veff(self, self.mol, self.make_rdm1())
    mf.__class__.get_veff = patched_get_veff

    original_intor_symmetric = pyscf.gto.Mole.intor_symmetric
    def patched_intor_symmetric(self, intor, *args, **kwargs):
        if 'ovlp' in intor and self.nao > 0:
            return np.eye(self.nao)
        return original_intor_symmetric(self, intor, *args, **kwargs)
    pyscf.gto.Mole.intor_symmetric = patched_intor_symmetric
    
    original_intor = pyscf.gto.Mole.intor
    def patched_intor(self, intor, *args, **kwargs):
        if 'ovlp' in intor and self.nao > 0:
            return np.eye(self.nao)
        return original_intor(self, intor, *args, **kwargs)
    pyscf.gto.Mole.intor = patched_intor

def main():
    start_timestamp = datetime.datetime.now()
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Input files (FCIDUMP, YAML config, QASM circuit) in any order")
    
    args = parser.parse_args()

    mol_file = None
    config_file = None
    custom_circuit_path = None
    
    for f in args.files:
        ext = os.path.splitext(f)[1].lower()
        if ext in [".yaml", ".yml"]:
            if config_file is not None and config_file != f:
                print(f"[Error] Multiple config files provided: {config_file} and {f}")
                return
            config_file = f
        elif ext in [".qasm", ".qpy"]:
            if custom_circuit_path is not None:
                print(f"[Error] Multiple custom circuits provided: {custom_circuit_path} and {f}")
                return
            custom_circuit_path = f
        else:
            if mol_file is not None:
                print(f"[Error] Multiple FCIDUMP files provided: {mol_file} and {f}")
                return
            mol_file = f
            
    if mol_file is None:
        print("[Error] No FCIDUMP file provided.")
        parser.print_help()
        return

    # Load config from YAML
    import config as run_config
    
    config_file = config_file or "config/default.yaml"
    
    if not os.path.exists(config_file):
        print(f"     [Info] Auto-generated default config at {config_file}")
        run_config.generate_default_yaml(config_file)
        
    config = run_config.load_config(config_file)

    if config.real_qpu and not config.backend:
        print("\n[Step 0] Discovering IBM Quantum Hardware...")
        try:
            from utils.ibm_utils import connect_backend
            qpu = connect_backend(
                use_real_qpu=True,
                specific_backend=None,
                min_qubits=config.max_qpu_orbitals
            )
            config.backend = qpu.name
            print(f"Dynamically locked to least busy backend: {config.backend}")
        except Exception as e:
            print(f"Warning: Failed to connect to IBM Quantum during discovery - {e}")

    # 1. Classical Prep
    print("\n[Step 1] PySCF Mean-Field Prep...")
    mf = tools.fcidump.to_scf(mol_file)
    mf.kernel()
    patch_abstract_orbitals(mf)
    
    norb = mf.mol.nao
    print(f"Full System Orbitals: {norb}")

    # 2. Smart Routing
    print("\n[Step 2] Routing Logic...")
    if norb <= config.classical_threshold:
        print(f"System ({norb} orbitals) <= Classical Threshold ({config.classical_threshold}).")
        print("Routing to Exact Classical Solver (CCSD)...")
        result = classical_solver.run(mf, config)
        total_energy = result.energy
        valid_ratio = result.metadata.get("valid_ratio", "100.0% (Exact Classical)")
    elif norb <= config.qpu_threshold:
        print(f"System ({norb} orbitals) <= Max QPU Capacity ({config.qpu_threshold}).")
        print("Routing directly to Quantum Solver (SPIRAL) without fragmentation...")
        result = quantum_solver.run(mf, config, custom_circuit_path=custom_circuit_path)
        total_energy = result.energy
        valid_ratio = result.metadata.get("valid_ratio", "N/A")
        
        result.metadata["total_fragments"] = 1
        result.metadata["quantum_fragments"] = 1
        result.metadata["classical_fragments"] = 0
        result.metadata["fallback_fragments"] = 0
        result.metadata["skipped_fragments"] = 0
    else:
        print(f"System ({norb} orbitals) > Max QPU Capacity ({config.qpu_threshold}).")
        print("Routing to EWF Fragment Orchestrator...")
        result = frag_manager.run(mf, config)
        total_energy = result.energy
        valid_ratio = result.metadata.get("valid_ratio", "N/A")

    # Output Results
    elapsed = time.perf_counter() - start_time
    formatted_duration = str(datetime.timedelta(seconds=int(elapsed)))
    
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    
    console = Console()

    WIDTH = 25
    
    # Meta Info
    meta_table = Table(show_header=False, box=None)
    meta_table.add_column("Key", width=WIDTH)
    meta_table.add_column("Value")
    meta_table.add_row("Started:", start_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
    meta_table.add_row("Time Elapsed:", formatted_duration)
    meta_table.add_row("Molecule System:", os.path.basename(mol_file))
    meta_table.add_row("Total Orbitals:", str(norb))
    meta_table.add_row("Injected Circuit:", os.path.basename(custom_circuit_path) if custom_circuit_path else "None")
    
    # Config Info
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Key", width=WIDTH)
    config_table.add_column("Value")
    
    # Routing
    config_table.add_row("QPU Threshold:", str(config.qpu_threshold))
    config_table.add_row("Classical Threshold:", str(config.classical_threshold))
    config_table.add_row("Classical FCI Threshold:", str(config.fci_threshold))
    
    # EWF
    config_table.add_row("EWF Bath Type:", config.ewf_bath_type)
    config_table.add_row("EWF Truncation:", f"{config.ewf_truncation:.1e}")
    
    # Auto-Tuning
    config_table.add_row("Num Layouts:", str(config.num_layouts))
    config_table.add_row("Top Candidates (Sim):", str(getattr(config, "tuning_top_candidates", 3)))
    config_table.add_row("Sim Qubit Threshold:", str(getattr(config, "tuning_sim_threshold", 20)))
    config_table.add_row("Beta (Fidelity):", str(config.beta))
    config_table.add_row("Min Shot Count:", str(config.min_count))
    
    # QPU
    config_table.add_row("SPIRAL Compilation:", "Enabled" if getattr(config, "use_spiral", True) else "Disabled")
    
    if getattr(config, "use_spanning_tree", False):
        spanning_status = "Enabled" if getattr(config, "connectivity", "heavy-hex") == "square" else "Enabled (Unused)"
    else:
        spanning_status = "Disabled"
    config_table.add_row("Spanning Tree:", spanning_status)
    
    config_table.add_row("Real QPU:", str(config.real_qpu))
    if config.real_qpu:
        backend_str = config.backend if config.backend else "IBM Quantum"
    else:
        backend_str = "Simulator"
    config_table.add_row("Target Backend:", backend_str)
    config_table.add_row("Shots:", str(config.shots))
    config_table.add_row("Connectivity:", config.connectivity)
    config_table.add_row("LUCJ Reps:", str(config.lucj_reps))
    config_table.add_row("LUCJ Tolerance:", f"{config.lucj_tol:.1e}")
    config_table.add_row("LUCJ Optimize:", str(config.lucj_optimize))
    config_table.add_row("LUCJ Method:", config.lucj_method)
    
    # TrimSQD
    config_table.add_row("Trim Fraction:", str(config.trim_fraction))
    config_table.add_row("Trim Stop Tolerance:", f"{config.trim_stop_tol:.1e}")
    
    # SPIRAL Compilation Settings
    config_table.add_row("SPIRAL Chunk Size:", str(config.spiral_chunk_size))
    config_table.add_row("SPIRAL Max Gates:", str(config.spiral_max_gates))
    
    # Reference / Grading
    config_table.add_row("Reference Threshold:", str(config.reference_threshold))
    config_table.add_row("Reference FCI Threshold:", str(config.reference_fci_threshold))
    
    # Results Info
    res_table = Table(show_header=False, box=None)
    res_table.add_column("Key", width=WIDTH)
    res_table.add_column("Value", style="bold")
    
    res_table.add_row("Valid Shot Ratio:", valid_ratio)
    
    if norb <= config.reference_threshold:
        try:
            from pyscf import cc, fci
            # Suppress CCSD printing for the report phase
            mycc = cc.CCSD(mf)
            mycc.verbose = 0
            mycc.kernel()
            if norb <= config.reference_fci_threshold:
                cisolver = fci.FCI(mf)
                cisolver.verbose = 0
                fci_energy, _ = cisolver.kernel()
                ref_label = "FCI Reference:"
            else:
                fci_energy = mycc.e_tot + mycc.ccsd_t()
                ref_label = "CCSD(T) Reference:"
            
            res_table.add_row(ref_label, f"{fci_energy:.8f} Ha")
            res_table.add_row("Total SQD Energy:", f"{total_energy:.8f} Ha")
            res_table.add_row("SQD Energy Error:", f"{abs(total_energy - fci_energy):.3e} Ha")
        except Exception as e:
            res_table.add_row("Total SQD Energy:", f"{total_energy:.8f} Ha")
            res_table.add_row("[Reference Error]:", str(e))
    else:
        res_table.add_row("Total SQD Energy:", f"{total_energy:.8f} Ha")
        res_table.add_row("Energy Reference:", "Skipped (reference threshold)")
        
    # Run Details Info (if fragmented)
    run_details_table = None
    if "total_fragments" in result.metadata:
        run_details_table = Table(show_header=False, box=None)
        run_details_table.add_column("Key", justify="left", width=20)
        run_details_table.add_column("Value")
        run_details_table.add_row("Total Fragments:", str(result.metadata["total_fragments"]))
        run_details_table.add_row(" -> Quantum:", str(result.metadata["quantum_fragments"]))
        run_details_table.add_row(" -> Classical:", str(result.metadata["classical_fragments"]))
        run_details_table.add_row(" -> QPU Fallback:", str(result.metadata["fallback_fragments"]))
        run_details_table.add_row(" -> Skipped:", str(result.metadata["skipped_fragments"]))
        
    q_table = None
    if "circuit_depth" in result.metadata:
        q_table = Table(show_header=False, box=None)
        q_table.add_column("Key", width=WIDTH)
        q_table.add_column("Value")
        q_table.add_row("Circuit Depth:", str(result.metadata.get("circuit_depth", "N/A")))
        two_q_label = f"{result.metadata.get('two_q_gate_name', 'CX')} Gates:"
        q_table.add_row(two_q_label, str(result.metadata.get("two_q_count", "N/A")))
        q_table.add_row("SWAP Gates:", str(result.metadata.get("swap_count", "N/A")))
        q_table.add_row("Top Bitstrings:", result.metadata.get("top_bitstrings", "N/A"))

    main_table = Table(show_edge=False, show_header=False, expand=False)
    main_table.add_column()
    main_table.add_row(Panel(meta_table, title="[bold]Summary"))
    main_table.add_row(Panel(config_table, title="[bold]Configurations"))
    if run_details_table:
        main_table.add_row(Panel(run_details_table, title="[bold]Execution"))
    if q_table:
        main_table.add_row(Panel(q_table, title="[bold]Quantum Execution"))
    main_table.add_row(Panel(res_table, title="[bold]Results"))
    
    console.print("\n")
    console.print(main_table)

if __name__ == "__main__":
    main()
