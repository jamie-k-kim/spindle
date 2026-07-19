fn main() {
    // Empty build script.
    // We no longer link against GAP via FFI because the legacy SPIRAL GAP engine
    // does not compile to a shared library (libgap.so).
    // We will interface with the GAP executable via subprocesses in Rust instead.
}
