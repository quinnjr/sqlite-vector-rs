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
    db.create_module("vector", module, ())?;
    scalar::register_scalar_functions(db)?;
    Ok(())
}

/// Register the extension on a rusqlite connection (library mode).
#[cfg(feature = "library")]
pub fn register(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    todo!("implement library-mode registration")
}
