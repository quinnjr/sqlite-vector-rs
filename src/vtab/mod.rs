pub mod config;
pub mod cursor;
pub mod shadow;
pub mod transaction;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use sqlite3_ext::query::ToParam;
use sqlite3_ext::vtab::{
    ChangeInfo, ChangeType, ConstraintOp, CreateVTab, DisconnectResult, FindFunctionVTab,
    IndexInfo, TransactionVTab, UpdateVTab, VTab, VTabConnection, VTabFunctionList,
};
use sqlite3_ext::{
    Error, FallibleIteratorMut, FromValue, Result, SQLITE_EMPTY, ValueRef, function::Context,
};

use crate::index::HnswIndex;
use crate::vtab::config::VectorTableConfig;
use crate::vtab::cursor::{CursorMode, VectorCursor};
use crate::vtab::shadow::ShadowOps;
use crate::vtab::transaction::{IndexState, VectorTransaction};

// Index numbers passed via best_index -> filter
const INDEX_SCAN: i32 = 0;
const INDEX_KNN: i32 = 1;

/// Default ANN candidate count when no LIMIT is safely consumable as k.
pub const DEFAULT_KNN_K: usize = 100;

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
fn save_meta_to_shadow(db: &VTabConnection, table_name: &str, meta_json: &str) -> Result<()> {
    let sql = ShadowOps::upsert_index_sql(table_name);
    db.execute(&sql, ["meta", meta_json])?;
    Ok(())
}

/// Load the persisted config `meta` row from the `_index` shadow table, if present.
fn load_meta_from_shadow(
    db: &VTabConnection,
    table_name: &str,
) -> Result<Option<serde_json::Value>> {
    let sql = ShadowOps::select_index_sql(table_name);
    match db.query_row(&sql, ["meta"], |row| Ok(row[0].get_str()?.to_owned())) {
        Ok(meta_json) => {
            let meta: serde_json::Value =
                serde_json::from_str(&meta_json).map_err(|e| Error::Module(e.to_string()))?;
            Ok(Some(meta))
        }
        Err(ref e) if *e == SQLITE_EMPTY => Ok(None),
        Err(e) => Err(e),
    }
}

/// Insert a new row into `_data` and return the auto-assigned rowid.
fn insert_into_data_shadow(
    db: &VTabConnection,
    config: &VectorTableConfig,
    explicit_id: Option<i64>,
    vector_blob: &[u8],
    metadata_args: &mut [&mut ValueRef],
) -> Result<i64> {
    use sqlite3_ext::query::Statement;
    let sql = match explicit_id {
        Some(_) => ShadowOps::insert_data_with_id_sql(config),
        None => ShadowOps::insert_data_sql(config),
    };
    db.insert(&sql, |stmt: &mut Statement| {
        let mut i = 1;
        if let Some(id) = explicit_id {
            id.bind_param(&mut *stmt, i)?;
            i += 1;
        }
        vector_blob.bind_param(&mut *stmt, i)?;
        i += 1;
        for val in metadata_args.iter_mut() {
            val.bind_param(&mut *stmt, i)?;
            i += 1;
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
// Registry — shared between the vtab module and the scalar functions on one
// connection, keyed by table name.
// ---------------------------------------------------------------------------

pub struct RegistryEntry {
    pub state: Weak<RefCell<IndexState>>,
    pub config: VectorTableConfig,
}

/// Shared between the vtab module and the scalar functions on one connection.
/// SQLite serializes all access on a connection; the Mutex satisfies Send
/// bounds, and the unsafe impls mirror VectorTable's single-thread invariant.
#[derive(Clone, Default)]
pub struct Registry(pub Arc<Mutex<HashMap<String, RegistryEntry>>>);
unsafe impl Send for Registry {}
unsafe impl Sync for Registry {}

impl Registry {
    /// Build the qualified `db.table` key used internally so that
    /// same-named vector tables in different attached databases (or main
    /// vs temp) don't overwrite each other's entry.
    fn qualified_key(config: &VectorTableConfig) -> String {
        format!("{}.{}", config.db_name, config.table_name)
    }

    pub fn register(&self, state: &Arc<RefCell<IndexState>>, config: &VectorTableConfig) {
        self.0.lock().unwrap().insert(
            Self::qualified_key(config),
            RegistryEntry {
                state: Arc::downgrade(state),
                config: config.clone(),
            },
        );
    }

    /// Look up a registered table by name. `name` may be a qualified
    /// `db.table` (exact match against the registry key) or a bare
    /// `table` (matched by suffix against `.table` across all registered
    /// entries). A bare name that matches more than one entry is
    /// ambiguous and returns an error rather than silently picking one.
    pub fn get(
        &self,
        name: &str,
    ) -> std::result::Result<(Arc<RefCell<IndexState>>, VectorTableConfig), String> {
        let map = self.0.lock().unwrap();

        if name.contains('.') {
            let e = map
                .get(name)
                .ok_or_else(|| format!("no vector table named {name}"))?;
            let state = e
                .state
                .upgrade()
                .ok_or_else(|| format!("no vector table named {name}"))?;
            return Ok((state, e.config.clone()));
        }

        let suffix = format!(".{name}");
        let mut matches: Vec<&RegistryEntry> = map
            .iter()
            .filter(|(k, _)| k.ends_with(&suffix))
            .map(|(_, e)| e)
            .collect();

        match matches.len() {
            0 => Err(format!("no vector table named {name}")),
            1 => {
                let e = matches.remove(0);
                let state = e
                    .state
                    .upgrade()
                    .ok_or_else(|| format!("no vector table named {name}"))?;
                Ok((state, e.config.clone()))
            }
            _ => Err(format!("ambiguous table name '{name}'; qualify as 'db.{name}'")),
        }
    }
}

// ---------------------------------------------------------------------------
// Reconcile: bring an in-memory HNSW index up to date with the `_data` shadow
// table after a fresh connect (or a rollback). Adds rows present in `_data`
// but missing from the graph; if the graph still holds stale keys (deletes
// that were never persisted), rebuilds it from scratch.
// ---------------------------------------------------------------------------

pub fn reconcile_index(
    db: &VTabConnection,
    config: &VectorTableConfig,
    index: HnswIndex,
) -> Result<HnswIndex> {
    let sql = ShadowOps::select_ids_vectors_sql(&config.table_name);
    let mut stmt = db.prepare(&sql)?;
    stmt.query(())?;
    let mut count: usize = 0;
    while let Some(row) = stmt.next()? {
        count += 1;
        let id = row[0].get_i64() as u64;
        if !index.contains(id) {
            index
                .add(id, row[1].get_blob()?)
                .map_err(|e| Error::Module(e.to_string()))?;
        }
    }
    if index.len() == count {
        return Ok(index);
    }
    // Graph holds keys that no longer exist in _data (deletes lost since the
    // last persist): rebuild from scratch.
    let fresh = HnswIndex::new(config.dim, config.vtype, config.metric, Some(config.hnsw_params))
        .map_err(|e| Error::Module(e.to_string()))?;
    let mut stmt = db.prepare(&sql)?;
    stmt.query(())?;
    while let Some(row) = stmt.next()? {
        fresh
            .add(row[0].get_i64() as u64, row[1].get_blob()?)
            .map_err(|e| Error::Module(e.to_string()))?;
    }
    Ok(fresh)
}

// ---------------------------------------------------------------------------
// Shared init logic used by both connect and create
// ---------------------------------------------------------------------------

fn init(
    db: &VTabConnection,
    args: &[&str],
    verify_against_meta: bool,
) -> Result<VectorTableConfig> {
    let mut config = VectorTableConfig::parse(args).map_err(|e| Error::Module(e.to_string()))?;

    // On connect (not create) reconcile the parsed args against the persisted
    // `meta` row: dim/type/metric must agree (they define the shape of the
    // shadow tables and the on-disk vectors), while m/ef_construction/ef_search
    // are HNSW tuning knobs where the persisted values always win, since the
    // caller may omit them on subsequent CREATE VIRTUAL TABLE (re)connects.
    if verify_against_meta
        && let Some(meta) = load_meta_from_shadow(db, &config.table_name)?
    {
        let (dim, vtype, metric, params) = VectorTableConfig::params_from_meta(&meta)
            .map_err(|e| Error::Module(e.to_string()))?;
        if dim != config.dim || vtype != config.vtype || metric != config.metric {
            return Err(Error::Module(format!(
                "declared parameters disagree with persisted meta for {}",
                config.table_name
            )));
        }
        config.hnsw_params = params;
    }

    Ok(config)
}

// ---------------------------------------------------------------------------
// Shared vtab construction used by both connect and create, after the shadow
// tables (and, for create, the meta row) already exist. Loads any persisted
// index, reconciles it against `_data`, registers it, and wires up the
// knn_match overload.
// ---------------------------------------------------------------------------

#[allow(clippy::arc_with_non_send_sync)]
fn build_vtab<'vtab>(
    db: &VTabConnection,
    aux: &Registry,
    config: VectorTableConfig,
) -> Result<(String, VectorTable<'vtab>)> {
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

    let index = reconcile_index(db, &config, index)?;

    let snapshot = index
        .save_to_buffer()
        .map_err(|e| Error::Module(e.to_string()))?;
    let state = Arc::new(RefCell::new(IndexState {
        index,
        dirty: false,
        last_committed: Some(snapshot),
        changes_since_persist: 0,
    }));

    aux.register(&state, &config);

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
    type Aux = Registry;
    type Cursor = VectorCursor;

    fn connect(
        db: &'vtab VTabConnection,
        aux: &'vtab Self::Aux,
        args: &[&str],
    ) -> Result<(String, Self)> {
        let config = init(db, args, true)?;
        build_vtab(db, aux, config)
    }

    fn best_index(&'vtab self, info: &mut IndexInfo) -> Result<()> {
        // Distance column index = 2 + num_metadata_cols
        let distance_col = (2 + self.config.metadata_columns.len()) as i32;

        // Pass 1: classify.
        let mut has_knn = false;
        let mut has_limit = false;
        let mut has_other = false;
        for c in info.constraints() {
            if !c.usable() {
                continue;
            }
            if c.column() == distance_col && matches!(c.op(), ConstraintOp::Function(_)) {
                has_knn = true;
            } else if matches!(c.op(), ConstraintOp::Limit) {
                has_limit = true;
            } else if !matches!(c.op(), ConstraintOp::Offset) {
                has_other = true;
            }
        }

        // ORDER BY is consumable iff absent or exactly `distance ASC`.
        let mut ob = info.order_by();
        let (first, second) = (ob.next(), ob.next());
        let has_order_by = first.is_some();
        let order_consumable = match (first, second) {
            (None, _) => true,
            (Some(o), None) => o.column() == distance_col && !o.desc(),
            _ => false,
        };

        let take_limit = has_knn && has_limit && order_consumable && !has_other;

        // Pass 2: assign argv slots. argv 0 = query blob; argv 1 = k (only when taken).
        let mut argv_next: u32 = 1;
        for mut c in info.constraints() {
            if !c.usable() {
                continue;
            }
            if c.column() == distance_col && matches!(c.op(), ConstraintOp::Function(_)) {
                c.set_argv_index(Some(argv_next - 1));
                c.set_omit(true);
                argv_next += 1;
            } else if matches!(c.op(), ConstraintOp::Limit) && take_limit {
                c.set_argv_index(Some(argv_next - 1));
                argv_next += 1;
            }
        }

        if has_knn {
            info.set_index_num(INDEX_KNN);
            if has_order_by && order_consumable {
                info.set_order_by_consumed(true);
            }
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
        let config = init(db, args, false)?;

        // Create the shadow tables before the index is loaded/reconciled, so
        // build_vtab's reconcile pass (and any future connect) sees them.
        db.execute(&ShadowOps::create_data_table_sql(&config), ())?;
        db.execute(&ShadowOps::create_index_table_sql(&config), ())?;

        // Persist the resolved config so a bare-name vector_rebuild_index(t)
        // and future connect()s can recover the real dim/type/metric/HNSW params.
        save_meta_to_shadow(db, &config.table_name, &config.to_meta_json())?;

        build_vtab(db, aux, config)
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
                {
                    let mut s = self.state.borrow_mut();
                    s.dirty = true;
                    s.changes_since_persist += 1;
                }
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
                let explicit_id = if !args[0].is_null() {
                    Some(args[0].get_i64())
                } else if !args[1].is_null() {
                    Some(args[1].get_i64())
                } else {
                    None
                };
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

                let rowid = insert_into_data_shadow(db, &self.config, explicit_id, &vector_blob, meta_args)?;

                let state = self.state.borrow();
                state
                    .index
                    .add(rowid as u64, &vector_blob)
                    .map_err(|e| Error::Module(e.to_string()))?;
                drop(state);
                {
                    let mut s = self.state.borrow_mut();
                    s.dirty = true;
                    s.changes_since_persist += 1;
                }

                Ok(rowid)
            }
            ChangeType::Update => {
                use sqlite3_ext::query::Statement;
                let old_rowid = info.rowid().get_i64();
                let args = info.args_mut();
                let num_meta = self.config.metadata_columns.len();
                // args[0] = new rowid hint, args[1] = id col value, args[2] = vector,
                // args[3..3+N] = metadata columns.
                //
                // `ValueRef::nochange()` is the correct discriminator between "column
                // untouched by this UPDATE" and "user explicitly set it" (see
                // sqlite3_value_nochange). Empirically (see task-1-report.md, Fix round
                // 1) this vtab's cursor never opts into the nochange optimization, so
                // nochange() is always false and SQLite backfills untouched columns
                // with their real old values instead of NULL/nochange sentinels. We
                // still branch on nochange() here: if it is false (the observed case)
                // this degrades to a full-column UPDATE, which is correct because the
                // "old" values SQLite supplied are the real unchanged data; if a future
                // SQLite/cursor change ever makes nochange() true for untouched
                // columns, this code already does the right thing (skips rebinding
                // that column and reuses the existing row.)
                let id_unchanged = args[1].nochange();
                let new_rowid = if id_unchanged {
                    old_rowid
                } else {
                    args[1].get_i64()
                };

                let vector_unchanged = args[2].nochange();
                let vector_blob: Option<Vec<u8>> = if vector_unchanged {
                    None
                } else if args[2].is_null() {
                    // Vector is NOT NULL at the schema level; a genuine (non-nochange)
                    // NULL means the statement tried to null out the vector column.
                    return Err(Error::Module(
                        "vector column cannot be NULL".to_string(),
                    ));
                } else {
                    let blob = args[2].get_blob()?.to_vec();
                    self.config
                        .vtype
                        .validate_finite(&blob, self.config.dim)
                        .map_err(|e| Error::Module(e.to_string()))?;
                    Some(blob)
                };

                let mut changed_meta_idx = Vec::with_capacity(num_meta);
                for (i, val) in args[3..3 + num_meta].iter().enumerate() {
                    if !val.nochange() {
                        changed_meta_idx.push(i);
                    }
                }

                let include_id = !id_unchanged;
                let include_vector = vector_blob.is_some();
                let rowid_changed = new_rowid != old_rowid;
                let needs_reindex = rowid_changed || include_vector;

                // If we need to re-key the index but the vector itself isn't
                // changing, fetch the existing vector BEFORE mutating the shadow
                // row (once the row's id changes, it's no longer reachable at
                // old_rowid).
                let reindex_vector: Option<Vec<u8>> = if needs_reindex && vector_blob.is_none() {
                    match fetch_row_from_shadow(db, &self.config, old_rowid)? {
                        Some((_, v)) => Some(v),
                        None => {
                            return Err(Error::Module(format!(
                                "Row {old_rowid} not found in shadow table"
                            )));
                        }
                    }
                } else {
                    None
                };

                if include_id || include_vector || !changed_meta_idx.is_empty() {
                    let sql = ShadowOps::update_data_sql(
                        &self.config,
                        include_id,
                        include_vector,
                        &changed_meta_idx,
                    );
                    let meta_args = &mut args[3..3 + num_meta];
                    db.execute(&sql, |stmt: &mut Statement| {
                        let mut pos = 1;
                        if include_id {
                            new_rowid.bind_param(&mut *stmt, pos)?;
                            pos += 1;
                        }
                        if let Some(v) = &vector_blob {
                            v.as_slice().bind_param(&mut *stmt, pos)?;
                            pos += 1;
                        }
                        for (i, val) in meta_args.iter_mut().enumerate() {
                            if changed_meta_idx.contains(&i) {
                                val.bind_param(&mut *stmt, pos)?;
                                pos += 1;
                            }
                        }
                        old_rowid.bind_param(&mut *stmt, pos)?;
                        Ok(())
                    })?;
                }

                if needs_reindex {
                    let final_vector = vector_blob
                        .or(reindex_vector)
                        .expect("vector available for reindex: computed above");
                    let state = self.state.borrow();
                    state
                        .index
                        .remove(old_rowid as u64)
                        .map_err(|e| Error::Module(e.to_string()))?;
                    state
                        .index
                        .add(new_rowid as u64, &final_vector)
                        .map_err(|e| Error::Module(e.to_string()))?;
                    drop(state);
                    {
                        let mut s = self.state.borrow_mut();
                        s.dirty = true;
                        s.changes_since_persist += 1;
                    }
                }

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
            snapshots: Vec::new(),
            sync_every: self.config.sync_every,
            config: self.config.clone(),
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
