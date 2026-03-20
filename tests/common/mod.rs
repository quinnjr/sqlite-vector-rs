use rusqlite::Connection;
use std::path::Path;

/// Create an in-memory SQLite connection with the vector extension loaded.
pub fn open_with_extension() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    unsafe {
        conn.load_extension_enable().unwrap();
        let ext_path = find_extension_path();
        conn.load_extension(ext_path, None::<&str>).unwrap();
        conn.load_extension_disable().unwrap();
    }
    conn
}

/// Create a file-backed SQLite connection (for persistence tests).
pub fn open_file_with_extension(path: &Path) -> Connection {
    let conn = Connection::open(path).unwrap();
    unsafe {
        conn.load_extension_enable().unwrap();
        let ext_path = find_extension_path();
        conn.load_extension(ext_path, None::<&str>).unwrap();
        conn.load_extension_disable().unwrap();
    }
    conn
}

fn find_extension_path() -> &'static str {
    if cfg!(target_os = "linux") {
        "target/debug/libsqlite_vector_rs"
    } else if cfg!(target_os = "macos") {
        "target/debug/libsqlite_vector_rs"
    } else {
        "target/debug/sqlite_vector_rs"
    }
}

/// Generate a random f32 vector of given dimension.
pub fn random_f32_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::rng();
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}
