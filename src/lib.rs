pub mod types;
pub mod json;
pub mod distance;
pub mod index;
pub mod vtab;
pub mod scalar;

#[cfg(feature = "loadable_extension")]
use sqlite3_ext::*;

/// Entry point for the loadable SQLite extension.
#[cfg(feature = "loadable_extension")]
#[sqlite3_ext_main(persistent)]
fn sqlite3_extension_init(db: &Connection) -> Result<()> {
    db.create_module("vector", sqlite3_ext::vtab::StandardModule::<vtab::VectorTable>::new(), ())?;
    scalar::register_scalar_functions(db)?;
    Ok(())
}

/// Register the extension on a rusqlite connection (library mode).
#[cfg(feature = "library")]
pub fn register(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    todo!("implement library-mode registration")
}
