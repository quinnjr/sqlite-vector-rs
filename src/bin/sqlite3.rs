//! Standalone sqlite3 CLI with the vector extension pre-loaded.
//!
//! Build with: `cargo build --features library --bin sqlite3`
//! Usage: `./target/debug/sqlite3 [database_file]`

fn main() {
    // TODO: implement sqlite3 REPL with vector extension auto-registered
    // This will use rusqlite's bundled SQLite and register the vector
    // module + scalar functions on startup, then provide an interactive
    // SQL prompt.
    eprintln!("sqlite3-vector: not yet implemented");
    std::process::exit(1);
}
