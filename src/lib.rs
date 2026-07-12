pub mod arrow_io;
pub mod distance;
pub mod index;
pub mod json;
pub mod scalar;
pub mod types;
pub mod vtab;

#[cfg(feature = "loadable_extension")]
use sqlite3_ext::*;

/// Entry point for the loadable SQLite extension.
#[cfg(feature = "loadable_extension")]
#[sqlite3_ext_main(persistent)]
fn sqlite3_extension_init(db: &Connection) -> Result<()> {
    use sqlite3_ext::vtab::Module;
    let module = sqlite3_ext::vtab::StandardModule::<vtab::VectorTable<'_>>::new()
        .with_update()
        .with_transactions()
        .with_find_function();
    let registry = vtab::Registry::default();
    db.create_module("vector", module, registry.clone())?;
    scalar::register_scalar_functions(db, registry)?;
    Ok(())
}

/// Register the vector extension on a rusqlite connection.
///
/// Loads the companion shared library (`libsqlite_vector_rs.so` / `.dylib` / `.dll`),
/// which provides the `vector` virtual table module and all scalar functions.
///
/// The shared library is located by searching (in order):
///
/// 1. The `SQLITE_VECTOR_RS_LIB` environment variable (path with or without extension)
/// 2. The directory containing the current executable
/// 3. `target/debug/` and `target/release/` relative to the working directory
///
/// # Errors
///
/// Returns an error if the shared library cannot be found or loaded.
///
/// # Example
///
/// ```no_run
/// let conn = rusqlite::Connection::open_in_memory().unwrap();
/// sqlite_vector_rs::register(&conn).unwrap();
/// conn.execute_batch("
///     CREATE VIRTUAL TABLE embeddings USING vector(
///         dim=3, type=float4, metric=cosine
///     );
/// ").unwrap();
/// ```
#[cfg(feature = "library")]
pub fn register(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    let path = find_extension_path().ok_or_else(|| {
        rusqlite::Error::ModuleError(
            "Could not find the sqlite-vector-rs extension library. \
             Build with `cargo build` first, or set SQLITE_VECTOR_RS_LIB to the library path."
                .into(),
        )
    })?;

    unsafe {
        conn.load_extension_enable()?;
    }

    let result = unsafe { conn.load_extension(&path, None::<&str>) };

    let _ = conn.load_extension_disable();

    result
}

#[cfg(feature = "library")]
fn find_extension_path() -> Option<String> {
    use std::path::Path;

    let stem = if cfg!(target_os = "windows") {
        "sqlite_vector_rs"
    } else {
        "libsqlite_vector_rs"
    };

    let extensions: &[&str] = if cfg!(target_os = "macos") {
        &[".dylib"]
    } else if cfg!(target_os = "windows") {
        &[".dll"]
    } else {
        &[".so"]
    };

    // 1. Explicit path via environment variable
    if let Ok(val) = std::env::var("SQLITE_VECTOR_RS_LIB") {
        if Path::new(&val).exists() {
            return Some(val);
        }
        if extensions
            .iter()
            .any(|ext| Path::new(&format!("{val}{ext}")).exists())
        {
            return Some(val);
        }
    }

    // 2. Adjacent to the current executable
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let base = dir.join(stem);
        let base_str = base.to_string_lossy();
        if extensions
            .iter()
            .any(|ext| Path::new(&format!("{base_str}{ext}")).exists())
        {
            return Some(base_str.into_owned());
        }
    }

    // 3. Cargo build output directories (development convenience)
    for profile in &["debug", "release"] {
        let base = format!("target/{profile}/{stem}");
        if extensions
            .iter()
            .any(|ext| Path::new(&format!("{base}{ext}")).exists())
        {
            return Some(base);
        }
    }

    None
}
