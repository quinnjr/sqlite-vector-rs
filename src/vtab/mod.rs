pub mod config;
pub mod cursor;
pub mod shadow;
pub mod transaction;

use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::query::ToParam;
use sqlite3_ext::vtab::{
    ChangeInfo, ChangeType, ConstraintOp, CreateVTab, DisconnectResult, FindFunctionVTab,
    IndexInfo, TransactionVTab, UpdateVTab, VTab, VTabConnection, VTabFunctionList,
};
use sqlite3_ext::{Error, FromValue, Result, SQLITE_EMPTY, ValueRef, function::Context};

use crate::index::HnswIndex;
use crate::vtab::config::VectorTableConfig;
use crate::vtab::cursor::{CursorMode, VectorCursor};
use crate::vtab::shadow::ShadowOps;
use crate::vtab::transaction::{IndexState, VectorTransaction};

// Index numbers passed via best_index -> filter
const INDEX_SCAN: i32 = 0;
const INDEX_KNN: i32 = 1;

/// The virtual table implementation for vector search.
///
/// `db` is a raw pointer to the VTabConnection that SQLite provides to connect/create.
/// SQLite guarantees the connection outlives the virtual table, so this pointer is valid
/// for the entire lifetime of VectorTable.
pub struct VectorTable<'vtab> {
    config: VectorTableConfig,
    state: Arc<RefCell<IndexState>>,
    /// Safety: valid for 'vtab lifetime — SQLite keeps the connection alive.
    db: *const VTabConnection,
    functions: VTabFunctionList<'vtab, Self>,
}

// Safety: VectorTable is only ever accessed from a single thread by SQLite's
// virtual table machinery.
unsafe impl Send for VectorTable<'_> {}
unsafe impl Sync for VectorTable<'_> {}

// ---------------------------------------------------------------------------
// Shadow table I/O stubs — wired up in Task 13
// ---------------------------------------------------------------------------

/// Load the serialized HNSW index blob from the `_index` shadow table, if present.
fn load_index_from_shadow(db: &VTabConnection, table_name: &str) -> Result<Option<Vec<u8>>> {
    let sql = ShadowOps::select_index_sql(table_name);
    match db.query_row(&sql, ["hnsw_graph"], |row| {
        let blob = row[0].get_blob()?;
        Ok(blob.to_vec())
    }) {
        Ok(buf) => Ok(Some(buf)),
        Err(ref e) if *e == SQLITE_EMPTY => Ok(None),
        Err(e) => Err(e),
    }
}

/// Persist schema/config metadata to the `_index` shadow table.
#[allow(dead_code)]
fn save_meta_to_shadow(db: &VTabConnection, table_name: &str, meta_json: &str) -> Result<()> {
    let sql = ShadowOps::upsert_index_sql(table_name);
    db.execute(&sql, ["meta", meta_json])?;
    Ok(())
}

/// Insert a new row into `_data` and return the auto-assigned rowid.
fn insert_into_data_shadow(
    db: &VTabConnection,
    config: &VectorTableConfig,
    vector_blob: &[u8],
    metadata_args: &mut [&mut ValueRef],
) -> Result<i64> {
    use sqlite3_ext::query::Statement;
    let sql = ShadowOps::insert_data_sql(config);
    db.insert(&sql, |stmt: &mut Statement| {
        vector_blob.bind_param(&mut *stmt, 1)?;
        for (i, val) in metadata_args.iter_mut().enumerate() {
            val.bind_param(&mut *stmt, (i + 2) as i32)?;
        }
        Ok(())
    })
}

/// Delete a row from `_data` by rowid.
fn delete_from_data_shadow(db: &VTabConnection, table_name: &str, rowid: i64) -> Result<()> {
    let sql = ShadowOps::delete_data_sql(table_name);
    db.execute(&sql, [rowid])?;
    Ok(())
}

/// Fetch a row from `_data` by rowid, returning (id, vector) or None if not found.
fn fetch_row_from_shadow(
    db: &VTabConnection,
    config: &VectorTableConfig,
    rowid: i64,
) -> Result<Option<(i64, Vec<u8>)>> {
    use sqlite3_ext::SQLITE_EMPTY;
    let sql = ShadowOps::select_data_sql(&config.table_name);
    match db.query_row(&sql, [rowid], |row| {
        let id = row[0].get_i64();
        let vector = row[1].get_blob()?.to_vec();
        Ok((id, vector))
    }) {
        Ok(pair) => Ok(Some(pair)),
        Err(ref e) if *e == SQLITE_EMPTY => Ok(None),
        Err(e) => Err(e),
    }
}


// ---------------------------------------------------------------------------
// Shared init logic used by both connect and create
// ---------------------------------------------------------------------------

#[allow(clippy::arc_with_non_send_sync)]
fn init<'vtab>(db: &VTabConnection, args: &[&str]) -> Result<(String, VectorTable<'vtab>)> {
    let config = VectorTableConfig::parse(args).map_err(|e| Error::Module(e.to_string()))?;

    let schema = config.vtab_schema();

    // Try to reload a previously persisted index; fall back to a fresh one.
    let index = match load_index_from_shadow(db, &config.table_name) {
        Ok(Some(buf)) => {
            let idx = HnswIndex::new(
                config.dim,
                config.vtype,
                config.metric,
                Some(config.hnsw_params),
            )
            .map_err(|e| Error::Module(e.to_string()))?;
            idx.load_from_buffer(&buf)
                .map_err(|e| Error::Module(e.to_string()))?;
            idx
        }
        _ => HnswIndex::new(
            config.dim,
            config.vtype,
            config.metric,
            Some(config.hnsw_params),
        )
        .map_err(|e| Error::Module(e.to_string()))?,
    };

    let state = Arc::new(RefCell::new(IndexState {
        index,
        dirty: false,
        last_committed: None,
    }));

    let functions = VTabFunctionList::default();
    // Register knn_match as a 2-arg overloaded function (col, param).
    // ConstraintOp::Function(0) tells best_index this function can act as a constraint.
    // The function body is a no-op returning 1 because set_omit(true) in best_index
    // prevents SQLite from evaluating it; the real work happens in filter().
    functions.add(
        2,
        "knn_match",
        Some(ConstraintOp::Function(150)),
        |ctx: &Context, _args: &mut [&mut ValueRef]| ctx.set_result(1i32),
    );

    let vtab = VectorTable {
        config,
        state,
        db: db as *const VTabConnection,
        functions,
    };

    Ok((schema, vtab))
}

// ---------------------------------------------------------------------------
// VTab impl
// ---------------------------------------------------------------------------

impl<'vtab> VTab<'vtab> for VectorTable<'vtab> {
    type Aux = ();
    type Cursor = VectorCursor;

    fn connect(
        db: &'vtab VTabConnection,
        _aux: &'vtab Self::Aux,
        args: &[&str],
    ) -> Result<(String, Self)> {
        init(db, args)
    }

    fn best_index(&'vtab self, info: &mut IndexInfo) -> Result<()> {
        // Distance column index = 2 + num_metadata_cols
        let distance_col = (2 + self.config.metadata_columns.len()) as i32;

        let mut found_knn = false;
        let mut argv_next: u32 = 1;

        for mut c in info.constraints() {
            if !c.usable() {
                continue;
            }
            if c.column() == distance_col
                && let ConstraintOp::Function(_) = c.op()
            {
                // knn_match(distance_col, query_blob): query_blob passed to filter
                c.set_argv_index(Some(argv_next - 1));
                c.set_omit(true);
                argv_next += 1;
                found_knn = true;
            }
            // Capture LIMIT as the k parameter for KNN searches
            if let ConstraintOp::Limit = c.op()
                && found_knn
            {
                c.set_argv_index(Some(argv_next - 1));
                c.set_omit(true);
                argv_next += 1;
            }
        }

        if found_knn {
            info.set_index_num(INDEX_KNN);
            info.set_estimated_cost(10.0);
            info.set_estimated_rows(10);
        } else {
            info.set_index_num(INDEX_SCAN);
            info.set_estimated_cost(1_000_000.0);
            info.set_estimated_rows(1_000_000);
        }

        Ok(())
    }

    fn open(&'vtab self) -> Result<Self::Cursor> {
        Ok(VectorCursor {
            mode: CursorMode::Scan {
                rows: Vec::new(),
                pos: 0,
            },
            num_metadata_cols: self.config.metadata_columns.len(),
            db: self.db,
            config: &self.config as *const VectorTableConfig,
            state: Arc::clone(&self.state),
        })
    }
}

// ---------------------------------------------------------------------------
// CreateVTab impl
// ---------------------------------------------------------------------------

impl<'vtab> CreateVTab<'vtab> for VectorTable<'vtab> {
    const SHADOW_NAMES: &'static [&'static str] = &["data", "index"];

    fn create(
        db: &'vtab VTabConnection,
        aux: &'vtab Self::Aux,
        args: &[&str],
    ) -> Result<(String, Self)> {
        let (schema, vtab) = init(db, args)?;

        // Create the shadow tables
        db.execute(&ShadowOps::create_data_table_sql(&vtab.config), ())?;
        db.execute(&ShadowOps::create_index_table_sql(&vtab.config), ())?;

        let _ = aux;
        Ok((schema, vtab))
    }

    fn destroy(self) -> DisconnectResult<Self> {
        // Safety: db pointer is valid for 'vtab; we're being destroyed now.
        let db = unsafe { &*self.db };
        for sql in ShadowOps::drop_shadow_tables_sql(&self.config.table_name) {
            if let Err(e) = db.execute(&sql, ()) {
                return Err((self, e));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UpdateVTab impl
// ---------------------------------------------------------------------------

impl<'vtab> UpdateVTab<'vtab> for VectorTable<'vtab> {
    fn update(&'vtab self, info: &mut ChangeInfo) -> Result<i64> {
        // Safety: db pointer is valid for 'vtab lifetime.
        let db = unsafe { &*self.db };

        match info.change_type() {
            ChangeType::Delete => {
                let rowid = info.rowid().get_i64();
                delete_from_data_shadow(db, &self.config.table_name, rowid)?;
                self.state
                    .borrow()
                    .index
                    .remove(rowid as u64)
                    .map_err(|e| Error::Module(e.to_string()))?;
                self.state.borrow_mut().dirty = true;
                Ok(0)
            }
            ChangeType::Insert => {
                let args = info.args_mut();
                // SQLite xUpdate argv layout (after argv[0] = old rowid):
                //   args[0] = new rowid (NULL → auto-assign)
                //   args[1] = col 0 (id)
                //   args[2] = col 1 (vector)
                //   args[3..3+N] = metadata cols
                //   args[3+N] = distance (hidden, ignored on insert)
                let vector_blob = args[2].get_blob()?.to_vec();
                let num_meta = self.config.metadata_columns.len();
                let meta_args = &mut args[3..3 + num_meta];

                // Validate dimension and finiteness before inserting
                self.config
                    .vtype
                    .validate_blob(&vector_blob, self.config.dim)
                    .map_err(|e| Error::Module(e.to_string()))?;
                self.config
                    .vtype
                    .validate_finite(&vector_blob, self.config.dim)
                    .map_err(|e| Error::Module(e.to_string()))?;

                let rowid = insert_into_data_shadow(db, &self.config, &vector_blob, meta_args)?;

                let state = self.state.borrow();
                state
                    .index
                    .add(rowid as u64, &vector_blob)
                    .map_err(|e| Error::Module(e.to_string()))?;
                drop(state);
                self.state.borrow_mut().dirty = true;

                Ok(rowid)
            }
            ChangeType::Update => {
                use sqlite3_ext::query::Statement;
                let old_rowid = info.rowid().get_i64();
                let args = info.args_mut();
                // args[0] = new rowid (NULL when unchanged), args[1] = id col value,
                // args[2] = vector, args[3..3+N] = metadata columns.
                // Always use args[1] (the id column value) as the authoritative source
                let new_rowid = if !args[1].is_null() {
                    args[1].get_i64()
                } else {
                    old_rowid
                };
                // If vector is NULL (not being updated), fetch the old vector from the database
                let vector_blob = if args[2].is_null() {
                    match fetch_row_from_shadow(db, &self.config, old_rowid)? {
                        Some(row) => row.1,
                        None => return Err(Error::Module(
                            format!("Row {} not found in shadow table", old_rowid),
                        )),
                    }
                } else {
                    args[2].get_blob()?.to_vec()
                };
                let num_meta = self.config.metadata_columns.len();

                self.config
                    .vtype
                    .validate_finite(&vector_blob, self.config.dim)
                    .map_err(|e| Error::Module(e.to_string()))?;

                let meta_args = &mut args[3..3 + num_meta];
                let sql = ShadowOps::update_data_sql(&self.config);
                db.execute(&sql, |stmt: &mut Statement| {
                    new_rowid.bind_param(&mut *stmt, 1)?;
                    vector_blob.as_slice().bind_param(&mut *stmt, 2)?;
                    let mut i = 3;
                    for val in meta_args.iter_mut() {
                        val.bind_param(&mut *stmt, i)?;
                        i += 1;
                    }
                    old_rowid.bind_param(&mut *stmt, i)?;
                    Ok(())
                })?;

                let state = self.state.borrow();
                state
                    .index
                    .remove(old_rowid as u64)
                    .map_err(|e| Error::Module(e.to_string()))?;
                state
                    .index
                    .add(new_rowid as u64, &vector_blob)
                    .map_err(|e| Error::Module(e.to_string()))?;
                drop(state);
                self.state.borrow_mut().dirty = true;

                Ok(new_rowid)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TransactionVTab impl
// ---------------------------------------------------------------------------

impl<'vtab> TransactionVTab<'vtab> for VectorTable<'vtab> {
    type Transaction = VectorTransaction;

    fn begin(&'vtab self) -> Result<Self::Transaction> {
        Ok(VectorTransaction {
            state: Arc::clone(&self.state),
            table_name: self.config.table_name.clone(),
            db: self.db,
        })
    }
}

// ---------------------------------------------------------------------------
// FindFunctionVTab impl
// ---------------------------------------------------------------------------

impl<'vtab> FindFunctionVTab<'vtab> for VectorTable<'vtab> {
    fn functions(&'vtab self) -> &'vtab VTabFunctionList<'vtab, Self> {
        &self.functions
    }
}
