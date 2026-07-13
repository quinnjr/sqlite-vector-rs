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
#[allow(dead_code)]
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
    if cfg!(target_os = "macos") {
        "target/debug/libsqlite_vector_rs"
    } else if cfg!(target_os = "windows") {
        "target/debug/sqlite_vector_rs"
    } else {
        "target/debug/libsqlite_vector_rs"
    }
}
