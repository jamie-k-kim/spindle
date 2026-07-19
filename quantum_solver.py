import psutil
import numpy as np
import ffsim
import rustworkx as rx
import concurrent.futures
from functools import partial

from qiskit import QuantumCircuit, QuantumRegister, qasm3
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager as runtime_pm
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

def get_candidate_layouts(backend, norb, connectivity, pairs_aa, pairs_ab, pairs_bb=None, limit=3):
    if pairs_bb is None: pairs_bb = pairs_aa
    backend_coupling_graph = _make_backend_cmap_pygraph(backend, 1.0, 0.10)
    layout_graph, allowed_pairs_ab = _get_layout_graph_and_allowed_pairs_ab(
        norb=norb, backend_coupling_graph=backend_coupling_graph,
        connectivity=connectivity, pairs_aa=pairs_aa, pairs_ab=pairs_ab, pairs_bb=pairs_bb
    )
    layouts = []
    if norb < 6:
        try:
            mappings = rx.vf2_mapping(backend_coupling_graph, layout_graph, subgraph=True, id_order=False, induced=False, call_limit=5000)
            unique_sets = []
            num_allowed = len(allowed_pairs_ab)
            for mapping in mappings:
                initial_layout = [-1] * (2 * norb + num_allowed)
                for key, value in mapping.items(): initial_layout[value] = key
                layout_cand = initial_layout[:-num_allowed]
                if -1 not in layout_cand:
                    q_set = frozenset(layout_cand)
                    if q_set not in unique_sets:
                        unique_sets.append(q_set)
                        layouts.append((layout_cand, allowed_pairs_ab))
                if len(layouts) >= limit: break
        except Exception: pass
        
    if len(layouts) == 0:
        layouts.append((None, allowed_pairs_ab))
    return layouts

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

def run(mf, config, t1=None, t2=None, norb=None, nelec=None, h1e=None, h2e=None, h0e=None) -> SolverResult:
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

    
    if t1 is None or t2 is None:
        print("     Running one-time classical CCSD for initial amplitudes...")
        c_res = classical_solver.run(mf, config, force_ccsd=True)
        t1 = c_res.metadata.get("t1")
        t2 = c_res.metadata.get("t2")
        if not config.use_spiral:
            c_res.metadata["valid_ratio"] = "100.0% (Exact Classical)"
            return c_res

    lucj_config = {"n_reps": config.lucj_reps, "optimize": config.lucj_optimize, "method": config.lucj_method}
    try:
        frag_qc = build_lucj_circuit(t1, t2, norb, nelec, config=lucj_config)
    except Exception as e:
        print(f"     Failed to build LUCJ circuit: {e}")
        return SolverResult(energy=0.0)

    # 2. Connect backend early
    backend = connect_backend(use_real_qpu=config.real_qpu, specific_backend=config.backend)

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
    if config.use_spiral and py_spiral_quantum is not None:
        spiral_gates = parse_circuit_to_spiral(frag_qc)
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
    
    # 6. ISA mapping
    print(f"     Mapping to physical ISA for {backend.name}...")
    try:
        pm = runtime_pm(backend=backend, optimization_level=1)
        isa_circuit = pm.run(transpiled_qc)
    except Exception as e:
        print(f"     ISA Mapping failed: {e}")
        return SolverResult(energy=0.0)

    # 7. Execute
    print(f"     Executing on backend...")
    result = run_circuit(isa_circuit, backend, shots=config.shots if config.shots > 0 else 1024)
    counts = get_counts(result)

    # Iterative TrimSQD
    if h1e is None:
        h1e = mf.get_hcore()
    if h2e is None:
        from pyscf import ao2mo
        h2e = ao2mo.restore(1, mf._eri, norb)
    if h0e is None:
        h0e = mf.mol.energy_nuc()

    print(f"     Running Iterative TrimSQD Error Mitigation...")
    e_trim, rdm1, rdm2, valid_ratio = iterative_trim_sqd(
        counts, h1e, h2e, norb, nelec, h0e, 
        trim_fraction=config.trim_fraction, stop_tol=config.trim_stop_tol
    )

    if e_trim is None:
        print("     TrimSQD failed to find valid subspace. Returning 0.0")
        return SolverResult(energy=0.0, metadata={"valid_ratio": f"{valid_ratio:.2%}"})

    return SolverResult(
        energy=e_trim,
        rdm1=rdm1,
        rdm2=rdm2,
        metadata={"valid_ratio": f"{valid_ratio:.2%}"}
    )
