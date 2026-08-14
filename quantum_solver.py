import os
import psutil
import numpy as np
import ffsim
import rustworkx as rx
from qiskit import QuantumCircuit, QuantumRegister, qasm3
from qiskit_ibm_runtime import SamplerV2 as Sampler

import classical_solver

from quantum_fragment_methods.application.solvers.base import SolverResult
from quantum_fragment_methods.application.solvers.quantum_zoo.utils.lucj import build_lucj_circuit

from utils.new_lucj_pass_manager import _make_backend_cmap_pygraph, _get_layout_graph_and_allowed_pairs_ab
from utils.ibm_utils import connect_backend, get_counts, run_circuit
from utils.sqd_utils import get_pyscf_ci_strings
from pyscf.fci import selected_ci

try:
    import py_spiral_quantum
except ImportError:
    py_spiral_quantum = None

def check_resources(num_qubits, num_gates):
    est_ram_gb = 0.1 + ((num_qubits ** 2) * num_gates) / 200000.0
    sys_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if est_ram_gb > sys_ram_gb:
        raise SystemError(
            f"INSUFFICIENT SYSTEM RESOURCES:\n"
            f"Compiling this configuration ({num_qubits} qubits, {num_gates} gates) requires an estimated {est_ram_gb:.1f} GB of RAM.\n"
            f"Your system only has {sys_ram_gb:.1f} GB of available RAM. Execution terminated."
        )

# VF2 call limit: matches the constant used in new_lucj_pass_manager for consistency.
_VF2_CALL_LIMIT = 30_000_000

def get_candidate_layouts(backend, norb, connectivity, pairs_aa, pairs_ab, pairs_bb=None, limit=3):
    if pairs_bb is None: pairs_bb = pairs_aa
    backend_coupling_graph = _make_backend_cmap_pygraph(backend, 1.0, 0.10)
    layout_graph, allowed_pairs_ab = _get_layout_graph_and_allowed_pairs_ab(
        norb=norb, backend_coupling_graph=backend_coupling_graph,
        connectivity=connectivity, pairs_aa=pairs_aa, pairs_ab=pairs_ab, pairs_bb=pairs_bb
    )
    virtual_edges = list(layout_graph.edge_list())
    layouts = []
    try:
        mappings = rx.vf2_mapping(
            backend_coupling_graph, layout_graph,
            subgraph=True, id_order=False, induced=False, call_limit=_VF2_CALL_LIMIT
        )
        unique_sets = []
        num_allowed = len(allowed_pairs_ab)
        for mapping in mappings:
            initial_layout = [-1] * (2 * norb + num_allowed)
            for key, value in mapping.items(): initial_layout[value] = key
            
            # Use the truncated layout just for uniqueness checking
            layout_cand = initial_layout[:-num_allowed] if num_allowed > 0 else initial_layout
            if -1 not in layout_cand:
                q_set = frozenset(layout_cand)
                if q_set not in unique_sets:
                    unique_sets.append(q_set)
                    # Yield the FULL layout so heuristic scorer can map ancilla edges, plus num_allowed to truncate later
                    layouts.append((initial_layout, num_allowed))
            if len(layouts) >= limit: break
    except Exception: pass

    if len(layouts) == 0:
        layouts.append((None, len(allowed_pairs_ab)))
    return layouts, virtual_edges

def parse_circuit_to_spiral(circuit):
    gates = []
    for instr in circuit.data:
        name = instr.operation.name.upper()
        q_indices = [circuit.find_bit(q).index for q in instr.qubits]
        if name in ["CX", "CNOT"]:
            gates.append({"type": "CNOT", "control": q_indices[0], "target": q_indices[1]})
        elif name in ["H", "X", "Y", "Z"]:
            gates.append({"type": name, "qubits": q_indices})
        elif name in ["CZ"]:
            gates.append({"type": "CZ", "control": q_indices[0], "target": q_indices[1]})
        elif name in ["RX", "RY", "RZ", "U", "rx", "ry", "rz", "u"]:
            params = [str(p) for p in instr.operation.params]
            gates.append({"type": name, "qubits": q_indices, "params": params})
        elif name in ["BARRIER", "MEASURE"]:
            continue
        elif len(q_indices) == 1:
            gates.append({"type": "U", "qubits": q_indices, "params": ["0.0", "0.0", "0.0"]})
    return gates

def iterative_trim_sqd(counts, h1e, h2e, norb, nelec, h0e, trim_fraction=0.10, stop_tol=1e-4):
    nela, nelb = nelec
    valid_sa, valid_sb = set(), set()
    valid_shots = 0
    total_shots = sum(counts.values())
    if total_shots == 0: total_shots = 1
    
    for bitstring, count in counts.items():
        a_int, b_int = get_pyscf_ci_strings(bitstring, norb, nela, nelb)
        if a_int is not None:
            valid_shots += count
            valid_sa.add(a_int)
            valid_sb.add(b_int)
            
    valid_ratio = valid_shots / total_shots
    if not valid_sa or not valid_sb:
        return None, None, None, valid_ratio

    valid_sa = sorted(list(valid_sa))
    valid_sb = sorted(list(valid_sb))
    
    sci_solver = selected_ci.SelectedCI()
    sci_solver.verbose = 0
    
    prev_energy = None
    prev_civec = None
    prev_sa = None
    prev_sb = None
    
    iteration = 0
    while True:
        iteration += 1
        ci_strs = (valid_sa, valid_sb)
        try:
            e_tot, civec = selected_ci.kernel_fixed_space(
                sci_solver, h1e, h2e, norb, nelec, ci_strs=ci_strs
            )
            e_tot += h0e
        except Exception as e:
            print(f"     TrimSQD iter {iteration} failed: {e}")
            break
            
        if prev_energy is not None:
            if e_tot > prev_energy + stop_tol:
                break
                
        prev_energy = e_tot
        prev_civec = civec
        prev_sa = valid_sa
        prev_sb = valid_sb
        
        alpha_weights = np.sum(civec**2, axis=1)
        beta_weights = np.sum(civec**2, axis=0)
        
        num_alpha_drop = max(1, int(len(valid_sa) * trim_fraction))
        num_beta_drop = max(1, int(len(valid_sb) * trim_fraction))
        
        if len(valid_sa) <= num_alpha_drop or len(valid_sb) <= num_beta_drop:
            break
            
        keep_alpha_idx = np.argsort(alpha_weights)[num_alpha_drop:]
        keep_beta_idx = np.argsort(beta_weights)[num_beta_drop:]
        
        valid_sa = [valid_sa[i] for i in keep_alpha_idx]
        valid_sb = [valid_sb[i] for i in keep_beta_idx]
        valid_sa.sort()
        valid_sb.sort()

    if prev_civec is None:
        return None, None, None, valid_ratio
        
    ci_strs = (prev_sa, prev_sb)
    try:
        rdm1, rdm2 = sci_solver.make_rdm12(prev_civec, norb, nelec, ci_strs=ci_strs)
    except Exception as e:
        print(f"     Failed to build RDMs: {e}")
        rdm1, rdm2 = None, None
        
    return prev_energy, rdm1, rdm2, valid_ratio

def run(mf, config, t1=None, t2=None, norb=None, nelec=None, h1e=None, h2e=None, h0e=None, backend=None, custom_circuit_path=None) -> SolverResult:
    """
    Run quantum compilation and execution.
    """
    if norb is None:
        norb = mf.mol.nao
    if nelec is None:
        try:
            nocc = mf.mol.nelec[0]
        except AttributeError:
            nocc = int(sum(mf.mo_occ) // 2)
        nelec = (nocc, nocc)

    
    # 1. Connect backend early so its topology can inform circuit generation
    if backend is None:
        backend = connect_backend(use_real_qpu=config.real_qpu, specific_backend=config.backend)

    # 1.5. Prepare Hamiltonian components early for auto-tuning scoring
    if h1e is None:
        h1e = mf.get_hcore()
    if h2e is None:
        from pyscf import ao2mo
        h2e = ao2mo.restore(1, mf._eri, norb)
    if h0e is None:
        h0e = mf.mol.energy_nuc()
        
    precomputed_interaction_pairs = None

    if custom_circuit_path and os.path.exists(custom_circuit_path):
        from qiskit import QuantumCircuit
        print(f"     [Info] Bypassing LUCJ. Loading custom circuit from {custom_circuit_path}...")
        if custom_circuit_path.endswith('.qpy'):
            from qiskit import qpy
            with open(custom_circuit_path, 'rb') as fd:
                frag_qc = qpy.load(fd)[0]
        else:
            frag_qc = QuantumCircuit.from_qasm_file(custom_circuit_path)
        if frag_qc.num_qubits != 2 * norb:
            print(f"     [Error] Custom circuit has {frag_qc.num_qubits} qubits, but the FCIDUMP describes a {2 * norb}-qubit system ({norb} spatial orbitals x 2 spins). These must match.")
            return SolverResult(energy=0.0)
        frag_qc.remove_final_measurements()
    else:
        if t1 is None or t2 is None:
            print("     Running one-time classical CCSD for initial amplitudes...")
            c_res = classical_solver.run(mf, config, force_ccsd=True)
            t1 = c_res.metadata.get("t1")
            t2 = c_res.metadata.get("t2")

        lucj_config = {"n_reps": getattr(config, "lucj_reps", 1), "optimize": getattr(config, "lucj_optimize", True), "method": getattr(config, "lucj_method", "L-BFGS-B")}
        use_spanning_tree = getattr(config, "use_spanning_tree", False)
        connectivity = getattr(config, "connectivity", "heavy-hex")

        # Pre-compute spanning tree interaction pairs once here so they can be shared
        # between build_lucj_circuit (ffsim ansatz) and SPIRAL's virtual_coupling,
        # avoiding a duplicate call and ensuring both always agree on the same topology.
        if use_spanning_tree and connectivity == "square":
            from utils.new_lucj_pass_manager import get_spanning_tree_interaction_pairs
            max_orbital_index = min(12, norb - 1)
            _pairs_aa = [(p, p + 1) for p in range(norb - 1)]
            _pairs_ab = [(p, p) for p in range(0, max_orbital_index + 1, 4)]
            precomputed_interaction_pairs = get_spanning_tree_interaction_pairs(
                backend=backend,
                norb=norb,
                connectivity="square",
                pairs_aa=_pairs_aa,
                pairs_ab=_pairs_ab,
                pairs_bb=_pairs_aa,
            )

        try:
            frag_qc = build_lucj_circuit(
                t1, t2, norb, nelec,
                config=lucj_config,
                backend=backend,
                connectivity=connectivity,
                use_spanning_tree=use_spanning_tree,
                precomputed_interaction_pairs=precomputed_interaction_pairs,
            )
        except Exception as e:
            print(f"     Failed to build LUCJ circuit: {e}")
            return SolverResult(energy=0.0)

        # 3. Decompose ffsim gates into basis gates
        try:
            from qiskit import transpile as qk_transpile
            frag_qc = ffsim.qiskit.PRE_INIT.run(frag_qc)
            frag_qc = qk_transpile(
                frag_qc,
                basis_gates=['u', 'cx', 'x', 'y', 'z', 'h', 'rx', 'ry', 'rz', 'cz'],
                optimization_level=1
            )
        except Exception as e:
            print(f"     Failed to unroll circuit: {e}")
            return SolverResult(energy=0.0)

    # 4. SPIRAL parse and compile
    if getattr(config, "use_spiral", False) and py_spiral_quantum is not None:
        spiral_gates = parse_circuit_to_spiral(frag_qc)
        
        # Full SABRE bypass: pass spanning tree edges to SPIRAL virtual_coupling.
        # Re-use precomputed_interaction_pairs from above to avoid a second VF2 call.
        if precomputed_interaction_pairs is not None:
            p_aa, p_ab, p_bb = precomputed_interaction_pairs
            virtual_coupling = (
                [(i, j) for i, j in p_aa] +
                [(i + norb, j + norb) for i, j in p_bb] +
                [(i, j + norb) for i, j in p_ab]
            )
        elif custom_circuit_path:
            edges = set()
            for inst in frag_qc.data:
                if len(inst.qubits) == 2:
                    q0 = frag_qc.find_bit(inst.qubits[0]).index
                    q1 = frag_qc.find_bit(inst.qubits[1]).index
                    edges.add((min(q0, q1), max(q0, q1)))
            virtual_coupling = list(edges)
            if not virtual_coupling:
                virtual_coupling = [(i, i + 1) for i in range(2 * norb - 1)]
        else:
            virtual_coupling = [(i, i + 1) for i in range(2 * norb - 1)]
        
        if not spiral_gates:
            print(f"     No significant excitations found (empty circuit). Skipping compilation.")
            return SolverResult(energy=0.0)

        print(f"     Compiling {len(spiral_gates)} parameterized gates via SPIRAL...")
        check_resources(2 * norb, len(spiral_gates))
        
        try:
            qasm3_template = py_spiral_quantum.compile_circuit(
                2 * norb, virtual_coupling, spiral_gates, config.spiral_chunk_size, config.spiral_max_gates
            )
            transpiled_qc = qasm3.loads(qasm3_template)
            cx_count = transpiled_qc.count_ops().get('cx', 0)
            print(f"     SPIRAL Compiled! Size: {transpiled_qc.num_qubits} qubits, {cx_count} CX gates.")
        except Exception as e:
            print(f"     SPIRAL compilation failed: {e}")
            return SolverResult(energy=0.0)
    else:
        print(f"     Skipping SPIRAL (py_spiral_quantum not found or --no-spiral passed).")
        transpiled_qc = frag_qc

    # 5. Measurements
    transpiled_qc.measure_all()
    
    # 6. ISA mapping & Auto-Tuning
    print(f"     Mapping to physical ISA for {backend.name}...")
    try:
        opt_level = getattr(config, "transpile_optimization_level", 1)
        
        if custom_circuit_path:
            print("     [Info] Custom circuit detected. Bypassing LUCJ auto-tuner. Using SABRE layout/routing.")
            from qiskit import transpile as qk_transpile
            isa_circuit = qk_transpile(transpiled_qc, backend=backend, optimization_level=opt_level)
            swap_count = isa_circuit.count_ops().get('swap', 0)
            print(f"     SABRE Routing complete. SWAP count: {swap_count}")
        else:
            from utils.new_lucj_pass_manager import generate_lucj_pass_manager
            max_orbital_index = min(12, norb - 1)
            pairs_aa = [(p, p + 1) for p in range(norb - 1)]
            pairs_ab = [(p, p) for p in range(0, max_orbital_index + 1, 4)]
        
            num_layouts = getattr(config, "num_layouts", 1)
            best_layout = None
        
            if num_layouts > 1:
                print(f"     Running pre-flight auto-tuning sweep with {num_layouts} candidate layouts...")
            
                # Setup the unwrapped calibration backend for heuristic scoring
                if getattr(config, "connectivity", "heavy-hex") == "square":
                    from qiskit.providers.fake_provider import GenericBackendV2
                    from qiskit.transpiler import CouplingMap
                    cmap = CouplingMap.from_grid(num_rows=4, num_columns=4)
                    calibration_backend = GenericBackendV2(num_qubits=16, coupling_map=cmap)
                else:
                    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
                    calibration_backend = FakeBrisbane()
                
                candidate_layouts, virtual_edges = get_candidate_layouts(
                    backend, norb, getattr(config, "connectivity", "heavy-hex"), pairs_aa, pairs_ab, limit=num_layouts
                )
            
                # --- Stage 1: Heuristic Pre-Filter ---
                from quantum_fragment_methods.application.solvers.quantum_zoo.utils.lucj import lightweight_layout_error_scoring
                from quantum_fragment_methods.application.solvers.quantum_zoo.utils.lucj import IBM_TWO_Q_GATES
            
                valid_candidates = [cand for cand in candidate_layouts if cand[0] is not None]
                valid_layouts = [cand[0] for cand in valid_candidates]
            
                try:
                    try:
                        two_q_gate_name = IBM_TWO_Q_GATES.intersection(calibration_backend.configuration().basis_gates).pop()
                    except:
                        two_q_gate_name = "cx"
                    scored = lightweight_layout_error_scoring(
                        backend=calibration_backend,
                        virtual_edges=virtual_edges,
                        physical_layouts=valid_layouts,
                        two_q_gate_name=two_q_gate_name
                    )
                except Exception:
                    # If heuristic scoring fails (e.g. GenericBackendV2 has no .properties()), fall back
                    scored = [[layout, float(i)] for i, layout in enumerate(valid_layouts)]
                
                tuning_top_candidates = getattr(config, "tuning_top_candidates", 3)
                top_scored = scored[:tuning_top_candidates]
            
                # Truncate ancillas for ISA mapping
                layout_to_num_allowed = {tuple(cand[0]): cand[1] for cand in valid_candidates}
                top_candidates = []
                for layout, _ in top_scored:
                    num_allowed = layout_to_num_allowed[tuple(layout)]
                    truncated_layout = layout[:-num_allowed] if num_allowed > 0 else layout
                    top_candidates.append(truncated_layout)
                
                # --- Stage 2: Simulation Tie-Breaker ---
                tuning_sim_threshold = getattr(config, "tuning_sim_threshold", 20)
            
                if (2 * norb) <= tuning_sim_threshold and tuning_sim_threshold > 0:
                    print(f"     Stage 2: Running simulation tie-breaker on top {len(top_candidates)} candidates...")
                    from qiskit_aer import AerSimulator
                    sweep_backend = AerSimulator.from_backend(calibration_backend)
                
                    best_score = float("inf")
                    for trunc_layout in top_candidates:
                        pm, _ = generate_lucj_pass_manager(
                            backend=backend,
                            norb=norb,
                            connectivity=getattr(config, "connectivity", "heavy-hex"),
                            interaction_pairs=(pairs_aa, pairs_ab, pairs_aa),
                            initial_layout=trunc_layout,
                            optimization_level=opt_level
                        )
                        sweep_circuit = pm.run(transpiled_qc)
                        sweep_result = run_circuit(sweep_circuit, sweep_backend, shots=10000)
                        sweep_counts = get_counts(sweep_result)
                    
                        e_trim, _, _, _ = iterative_trim_sqd(
                            sweep_counts, h1e, h2e, norb, nelec, h0e, 
                            trim_fraction=getattr(config, "trim_fraction", 0.10), 
                            stop_tol=getattr(config, "trim_stop_tol", 1e-4)
                        )
                    
                        if e_trim is not None and e_trim < best_score:
                            best_score = e_trim
                            best_layout = trunc_layout
                
                    print(f"     Best auto-tuned layout energy: {best_score}")
                else:
                    print(f"     Circuit too large for simulation (2*norb={2*norb} > {tuning_sim_threshold}). Bypassing Stage 2.")
                    best_layout = top_candidates[0]
            else:
                from quantum_fragment_methods.application.solvers.quantum_zoo.utils.lucj import get_zigzag_physical_layout
                best_layout, _ = get_zigzag_physical_layout(
                    norb, backend, score_layouts=True, connectivity=getattr(config, "connectivity", "heavy-hex")
                )
            
        if not custom_circuit_path:
            pm, _ = generate_lucj_pass_manager(
                backend=backend,
                norb=norb,
                connectivity=getattr(config, "connectivity", "heavy-hex"),
                interaction_pairs=(pairs_aa, pairs_ab, pairs_aa),
                initial_layout=best_layout,
                optimization_level=opt_level
            )
            isa_circuit = pm.run(transpiled_qc)
            swap_count = isa_circuit.count_ops().get('swap', 0)
            print(f"     ISA Circuit generated. SWAP count: {swap_count}")
    except Exception as e:
        print(f"     ISA Mapping failed: {e}")
        return SolverResult(energy=0.0)

    # 7. Execute
    print(f"     Executing on backend...")
    result = run_circuit(isa_circuit, backend, shots=getattr(config, "shots", 1024) if getattr(config, "shots", 1024) > 0 else 1024)
    counts = get_counts(result)

    print(f"     Running Iterative TrimSQD Error Mitigation...")
    e_trim, rdm1, rdm2, valid_ratio = iterative_trim_sqd(
        counts, h1e, h2e, norb, nelec, h0e, 
        trim_fraction=config.trim_fraction, stop_tol=config.trim_stop_tol
    )

    total_shots = sum(counts.values()) if counts else 1
    top_3 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
    top_strings = ", ".join([f"|{k}>: {v/total_shots*100:.1f}%" for k, v in top_3])

    # Detect the backend's native 2-qubit gate (ECR on real IBM hardware, CX on simulators)
    isa_ops = isa_circuit.count_ops()
    two_q_count = isa_ops.get('ecr', 0) or isa_ops.get('cx', 0) or isa_ops.get('cz', 0)
    two_q_gate_name = 'ECR' if isa_ops.get('ecr', 0) else ('CZ' if isa_ops.get('cz', 0) else 'CX')

    meta = {
        "valid_ratio": f"{valid_ratio:.2%}",
        "circuit_depth": isa_circuit.depth(),
        "two_q_count": two_q_count,
        "two_q_gate_name": two_q_gate_name,
        "swap_count": swap_count,
        "top_bitstrings": top_strings
    }

    if e_trim is None:
        print("     TrimSQD failed to find valid subspace. Returning 0.0")
        return SolverResult(energy=0.0, metadata=meta)

    return SolverResult(
        energy=e_trim,
        rdm1=rdm1,
        rdm2=rdm2,
        metadata=meta
    )
