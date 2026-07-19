#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::process::{Command, Stdio};
use std::fs;
use std::io::Write;
use std::env;

mod qasm_unparser;

#[pymodule]
fn py_spiral_quantum(_py: Python, m: &PyModule) -> PyResult<()> {
    
    #[pyfn(m)]
    #[pyo3(name = "initialize_gap")]
    fn initialize_gap() -> PyResult<()> {
        println!("Rust/PyO3: Using Subprocess interface for GAP.");
        Ok(())
    }

    #[pyfn(m)]
    #[pyo3(name = "compile_circuit")]
    fn compile_circuit(
        _py: Python,
        num_qubits: usize,
        coupling_map: Vec<(usize, usize)>,
        _gates: Vec<&PyDict>,
        chunk_size: usize,
        max_gates: usize
    ) -> PyResult<String> {
        
        let spiral_root = match fs::canonicalize("./spiral-software") {
            Ok(p) => p,
            Err(_) => match fs::canonicalize("../spiral-software") {
                Ok(p2) => p2,
                Err(_) => return Err(pyo3::exceptions::PyRuntimeError::new_err("Could not find spiral-software directory")),
            }
        };
        
        // 1. Build adjacency matrix
        let mut arch = vec![vec![0; num_qubits]; num_qubits];
        for (u, v) in coupling_map {
            arch[u][v] = 1;
            arch[v][u] = 1;
        }
        
        let mut arch_rows = Vec::new();
        for row in arch {
            let row_strs: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            arch_rows.push(format!("[ {} ]", row_strs.join(", ")));
        }
        let arch_str = format!("[ {} ]", arch_rows.join(", "));
        
        let mut ops = Vec::new();
        for g in _gates {
            let typ_any = match g.get_item("type") {
                Ok(Some(v)) => v,
                _ => continue,
            };
            let typ: String = typ_any.extract().unwrap_or_default();
            let typ_lower = typ.to_lowercase();
            
            let qubits_any = match g.get_item("qubits") {
                    Ok(Some(v)) => v,
                    _ => continue,
                };
                let qubits: Vec<usize> = qubits_any.extract().unwrap();
                
                if typ_lower == "rx" {
                    let params: Vec<String> = g.get_item("params").unwrap().unwrap().extract().unwrap();
                    ops.push(format!("[[{}], qRxT(1, {})]", qubits[0], params[0]));
                } else if typ_lower == "ry" {
                    let params: Vec<String> = g.get_item("params").unwrap().unwrap().extract().unwrap();
                    ops.push(format!("[[{}], qRyT(1, {})]", qubits[0], params[0]));
                } else if typ_lower == "rz" {
                    let params: Vec<String> = g.get_item("params").unwrap().unwrap().extract().unwrap();
                    ops.push(format!("[[{}], qRzT(1, {})]", qubits[0], params[0]));
                } else if typ_lower == "u" {
                    let params: Vec<String> = g.get_item("params").unwrap().unwrap().extract().unwrap();
                    ops.push(format!("[[{}], qRzT(1, {})]", qubits[0], params[1]));
                    ops.push(format!("[[{}], qRyT(1, {})]", qubits[0], params[0]));
                    ops.push(format!("[[{}], qRzT(1, {})]", qubits[0], params[2]));
                } else if typ_lower == "h" {
                    ops.push(format!("[[{}], qHT(1)]", qubits[0]));
                } else if typ_lower == "x" {
                    ops.push(format!("[[{}], qXT(1)]", qubits[0]));
                } else if typ_lower == "y" {
                    ops.push(format!("[[{}], qYT(1)]", qubits[0]));
                } else if typ_lower == "s" {
                    ops.push(format!("[[{}], qST(1, 1)]", qubits[0]));
                } else if typ_lower == "t" {
                    ops.push(format!("[[{}], qTT(1, 1)]", qubits[0]));
                } else if typ_lower == "z" {
                    ops.push(format!("[[{}], qZT(1)]", qubits[0]));
                } else if typ_lower == "cx" || typ_lower == "cnot" {
                    ops.push(format!("[[{}, {}], qCNOT(1, 0, arch)]", qubits[0], qubits[1]));
                }
        }
        let mut ops_chunks_lines = Vec::new();
        ops_chunks_lines.push("gates := [];;".to_string());
        for chunk in ops.chunks(250) {
            let chunk_str = format!("[ {} ]", chunk.join(", "));
            ops_chunks_lines.push(format!("Append(gates, {});;", chunk_str));
        }
        let ops_setup_str = ops_chunks_lines.join("\n");

        // Gate count limit: RandomRuleTree + QuantumRewrite is linear-time;
        if ops.len() > max_gates {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Circuit too large for SPIRAL optimization ({} gates > {} limit). Falling back to Qiskit.",
                ops.len(), max_gates
            )));
        }

        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let qasm_path = spiral_root.join(format!("qspiralout_{}_{}.qasm", pid, id));
        let script_path = spiral_root.join(format!("temp_compile_{}_{}.g", pid, id));

        let raw_path = spiral_root.join(format!("qspiralout_{}_{}", pid, id));

        // 3. Create the temporary GAP script
        let script_content = format!(
            "Load(quantum);;\n\
             Import(quantum);;\n\
             opts := SpiralDefaults;;\n\
             arch := {};;\n\
             {}\n\
             Print(\"CIRCUIT_BUILT_SUCCESSFULLY\\n\");\n\
             chunks := [];;\n\
             chunk_size := {};;\n\
             i := 1;;\n\
             while i <= Length(gates) do\n\
                 sub_gates := gates{{[i .. Minimum(i + chunk_size - 1, Length(gates))]}};\n\
                 sub_c := qCirc(arch, {}, sub_gates);\n\
                 sub_c := QuantumRewrite(SPLRuleTree(RandomRuleTree(sub_c, opts)), opts);\n\
                 Add(chunks, sub_c);\n\
                 i := i + chunk_size;\n\
             od;\n\
             circ := ApplyFunc(Compose, chunks);;\n\
\n\
             UnparseQASMToFile := function(spl, outfile)\n\
                 local cmd;\n\
                 PrintTo(outfile, spl);\n\
                 cmd := Concatenation(\"python3 ./namespaces/packages/quantum/unparser/unparser.py \", outfile);\n\
                 Exec(cmd);\n\
             end;;\n\
             UnparseQASMToFile(circ, \"{}\");;\n\
             quit;\n",
             arch_str, ops_setup_str, chunk_size, num_qubits, raw_path.display()
        );
        fs::write(&script_path, script_content).expect("Failed to write temporary GAP script");

        // 4. Run GAP
        let output = std::process::Command::new("sh")
            .current_dir(&spiral_root)
            .arg("-c")
            .arg(format!("bin/spiral < \"{}\"", script_path.display()))
            .output().expect("Failed to execute command");

        // let _ = fs::remove_file(&script_path);

        // 5. Read the output QASM
        if qasm_path.exists() {
            let qasm_content = fs::read_to_string(&qasm_path).expect("Failed to read QASM output");
            let _ = fs::remove_file(&qasm_path);
            let _ = fs::remove_file(&raw_path);
            Ok(qasm_content)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!("GAP compilation failed: no QASM output produced.\nSTDOUT:\n{}\nSTDERR:\n{}", stdout, stderr)))
        }
    }

    Ok(())
}
