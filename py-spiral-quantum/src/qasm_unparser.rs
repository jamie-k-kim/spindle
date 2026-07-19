//! OpenQASM 3.0 Unparser for SPIRAL quantum IR
//! 
//! This module converts the tensor AST produced by SPIRAL (passed via GAP)
//! into valid OpenQASM 3.0 source code with parameter support.

use std::fmt::Write;

/// A simplified AST representation of a quantum circuit operation.
/// In a real integration, this would be derived by traversing the GAP C structures.
pub enum QIRNode {
    /// e.g. Rz(theta) q[0];
    ParameterizedGate {
        name: String,
        parameter: String,
        target_qubits: Vec<usize>,
    },
    /// e.g. cx q[0], q[1];
    FixedGate {
        name: String,
        target_qubits: Vec<usize>,
    },
    /// Sequential composition of nodes
    Compose(Vec<QIRNode>),
}

pub fn unparse_qasm3(ast: &QIRNode, num_qubits: usize) -> Result<String, std::fmt::Error> {
    let mut out = String::new();
    
    // QASM 3 Header
    writeln!(&mut out, "OPENQASM 3.0;")?;
    writeln!(&mut out, "include \"stdgates.inc\";")?;
    
    // Declare the generic angle parameter. For variational ansatzes, 
    // we often use an array of parameters or individual named parameters.
    writeln!(&mut out, "input angle theta;")?;
    
    // Declare the quantum register
    writeln!(&mut out, "qubit[{}] q;\n", num_qubits)?;
    
    // Recursively unparse the AST
    unparse_node(ast, &mut out)?;
    
    Ok(out)
}

fn unparse_node(node: &QIRNode, out: &mut String) -> Result<(), std::fmt::Error> {
    match node {
        QIRNode::Compose(nodes) => {
            for n in nodes {
                unparse_node(n, out)?;
            }
        },
        QIRNode::FixedGate { name, target_qubits } => {
            let qubits_str = target_qubits.iter()
                .map(|q| format!("q[{}]", q))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "{} {};", name.to_lowercase(), qubits_str)?;
        },
        QIRNode::ParameterizedGate { name, parameter, target_qubits } => {
            let qubits_str = target_qubits.iter()
                .map(|q| format!("q[{}]", q))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(out, "{}({}) {};", name.to_lowercase(), parameter, qubits_str)?;
        }
    }
    Ok(())
}
