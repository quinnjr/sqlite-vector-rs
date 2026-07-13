pub mod config;
pub mod cursor;
pub mod shadow;
pub mod transaction;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use sqlite3_ext::query::{Statement, ToParam};
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
pub(crate) const DEFAULT_KNN_K: usize = 100;

/// Shadow-table SQL strings built once per table (at connect/create time)
/// instead of `format!`-ed on every operation. `update` is intentionally
/// absent here: `ShadowOps::update_data_sql` varies per-call on which
/// columns actually changed (id/vector/metadata), so it cannot be
/// prebuilt — the `Update` arm of `UpdateVTab::update` still calls
/// `ShadowOps::update_data_sql` directly.
pub(crate) struct TableSql {
    pub(crate) insert: String,
    pub(crate) insert_with_id: String,
    pub(crate) delete: String,
    pub(crate) fetch_by_id: String,
    pub(crate) scan_all: String,
    pub(crate) ids_vectors: String,
}

impl TableSql {
    pub(crate) fn new(config: &VectorTableConfig) -> Self {
        Self {
            insert: ShadowOps::insert_data_sql(config),
            insert_with_id: ShadowOps::insert_data_with_id_sql(config),
            delete: ShadowOps::delete_data_sql(&config.table_name),
            fetch_by_id: ShadowOps::select_data_sql(&config.table_name),
            scan_all: ShadowOps::select_all_data_sql(&config.table_name),
            ids_vectors: ShadowOps::select_ids_vectors_sql(&config.table_name),
        }
    }
}

/// The virtual table implementation for vector search.
///
/// `db` is a raw pointer to the VTabConnection that SQLite provides to connect/create.
/// SQLite guarantees the connection outlives the virtual table, so this pointer is valid
/// for the entire lifetime of VectorTable.
pub struct VectorTable<'vtab> {
    config: VectorTableConfig,
    sql: TableSql,
    state: Arc<RefCell<IndexState>>,
    /// Safety: valid for 'vtab lifetime — SQLite keeps the connection alive.
    db: *const VTabConnection,
    functions: VTabFunctionList<'vtab, Self>,
    /// Cloned handle to the same registry the table registered itself in at
    /// connect/create time. `disconnect()` doesn't receive `Aux`, so this is
    /// how it can remove its own entry.
    registry: Registry,
    /// Cached prepared statements for the `xUpdate` paths: insert,
    /// insert-with-explicit-id, delete-by-rowid, fetch-by-id. `Statement` is
    /// an owned struct with no connection lifetime,
    /// and `Statement::query`/`execute`/`insert` reset the statement and
    /// clear/rebind parameters on every call, so holding one across `update()`
    /// invocations and reusing it is safe as long as it's never touched from
    /// more than one call at a time — guaranteed by the same single-thread
    /// invariant documented on the `unsafe impl Send/Sync` below. The dynamic
    /// UPDATE SET path is intentionally excluded: its SQL text varies per
    /// call (see `TableSql`'s doc comment), so there is nothing stable to
    /// cache. The cursor's scan/KNN statements are out of scope here — it
    /// only holds raw `*const` pointers, not owned state, so caching would
    /// need a lifetime story of its own.
    stmt_insert: RefCell<Option<Statement>>,
    stmt_insert_with_id: RefCell<Option<Statement>>,
    stmt_delete: RefCell<Option<Statement>>,
    stmt_fetch_by_id: RefCell<Option<Statement>>,
}

// Safety: VectorTable is only ever accessed from a single thread by SQLite's
// virtual table machinery. This covers the raw `db` pointer as well as the
// cached `Statement`s above (each wraps a live `sqlite3_stmt*`): all are
// created and used exclusively under SQLite's one-thread-per-connection
// guarantee, so no cross-thread aliasing of the underlying sqlite3_stmt can
// occur.
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

/// Load the persisted config `meta` row from the `_index` shadow table, if
/// present. Shared by `init()` (connect-time verification), and by the
/// `vector_rebuild_index`/`vector_ef_search` scalar functions in
/// `src/scalar.rs` — do not duplicate this select+parse elsewhere.
pub(crate) fn load_meta_from_shadow(
    db: &sqlite3_ext::Connection,
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

/// Fetch the cached `Statement` from `cell`, preparing it against `sql` the
/// first time it's needed. Subsequent calls reuse the same prepared
/// statement — `Statement::query`/`execute`/`insert` reset it and rebind
/// parameters on every call (see the doc comment on `VectorTable`'s
/// `stmt_*` fields), so this is safe to call repeatedly across `xUpdate`
/// invocations.
fn cached_stmt<'a>(
    cell: &'a RefCell<Option<Statement>>,
    db: &VTabConnection,
    sql: &str,
) -> Result<std::cell::RefMut<'a, Option<Statement>>> {
    if cell.borrow().is_none() {
        let stmt = db.prepare(sql)?;
        *cell.borrow_mut() = Some(stmt);
    }
    Ok(cell.borrow_mut())
}

/// Insert a new row into `_data` and return the auto-assigned rowid.
fn insert_into_data_shadow(
    stmt: &mut Statement,
    explicit_id: Option<i64>,
    vector_blob: &[u8],
    metadata_args: &mut [&mut ValueRef],
) -> Result<i64> {
    stmt.insert(|stmt: &mut Statement| {
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
fn delete_from_data_shadow(stmt: &mut Statement, rowid: i64) -> Result<()> {
    stmt.execute([rowid])?;
    Ok(())
}

/// Fetch a row from `_data` by rowid, returning (id, vector) or None if not found.
fn fetch_row_from_shadow(stmt: &mut Statement, rowid: i64) -> Result<Option<(i64, Vec<u8>)>> {
    use sqlite3_ext::SQLITE_EMPTY;
    match stmt.query_row([rowid], |row| {
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

pub(crate) struct RegistryEntry {
    pub(crate) state: Weak<RefCell<IndexState>>,
    pub(crate) config: VectorTableConfig,
}

/// Shared between the vtab module and the scalar functions on one connection.
/// SQLite serializes all access on a connection; the Mutex satisfies Send
/// bounds, and the unsafe impls mirror VectorTable's single-thread invariant.
#[derive(Clone, Default)]
pub struct Registry(pub(crate) Arc<Mutex<HashMap<String, RegistryEntry>>>);
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

    /// Remove the exact `db.table` entry, if present. Called from both
    /// `destroy()` (DROP TABLE) and the vtab's `disconnect()` path so dead
    /// entries never linger to participate in bare-name suffix matching
    /// (a dropped `main.t` must not make a freshly created `aux.t` look
    /// "ambiguous" under the bare name `t`) and so long-lived connections
    /// don't grow the map unboundedly.
    pub fn unregister(&self, db_name: &str, table_name: &str) {
        let key = format!("{db_name}.{table_name}");
        self.0.lock().unwrap().remove(&key);
    }

    /// Look up a registered table by name. `name` may be a qualified
    /// `db.table` (exact match against the registry key) or a bare
    /// `table` (matched by suffix against `.table` across all registered
    /// entries). A bare name that matches more than one entry is
    /// ambiguous and returns an error rather than silently picking one.
    ///
    /// Entries whose `Weak` can no longer be upgraded (the vtab was
    /// disconnected/destroyed without going through `unregister`, e.g. an
    /// older sqlite3_ext version that never called our disconnect hook) are
    /// treated as absent for matching purposes, and are opportunistically
    /// pruned from the map while the lock is held. This is defense-in-depth
    /// on top of the explicit `unregister` calls, not a substitute for them.
    pub fn get(
        &self,
        name: &str,
    ) -> std::result::Result<(Arc<RefCell<IndexState>>, VectorTableConfig), String> {
        let mut map = self.0.lock().unwrap();

        if name.contains('.') {
            let live = match map.get(name) {
                Some(e) => e.state.upgrade().map(|state| (state, e.config.clone())),
                None => None,
            };
            return match live {
                Some(pair) => Ok(pair),
                None => {
                    map.remove(name);
                    Err(format!("no vector table named {name}"))
                }
            };
        }

        let suffix = format!(".{name}");
        let mut dead_keys: Vec<String> = Vec::new();
        let mut matches: Vec<(Arc<RefCell<IndexState>>, VectorTableConfig)> = Vec::new();
        for (k, e) in map.iter() {
            if !k.ends_with(&suffix) {
                continue;
            }
            match e.state.upgrade() {
                Some(state) => matches.push((state, e.config.clone())),
                None => dead_keys.push(k.clone()),
            }
        }
        for k in dead_keys {
            map.remove(&k);
        }

        match matches.len() {
            0 => Err(format!("no vector table named {name}")),
            1 => Ok(matches.remove(0)),
            _ => Err(format!(
                "ambiguous table name '{name}'; qualify as 'db.{name}'"
            )),
        }
    }
}

/// Map a constraint op to the wire token used in `best_index`'s filter spec
/// and argv-assignment passes. Kept as a single source of truth so the two
/// passes can never drift out of sync with each other.
fn op_token(op: ConstraintOp) -> Option<&'static str> {
    match op {
        ConstraintOp::Eq => Some("eq"),
        ConstraintOp::GT => Some("gt"),
        ConstraintOp::GE => Some("ge"),
        ConstraintOp::LT => Some("lt"),
        ConstraintOp::LE => Some("le"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Reconcile: bring an in-memory HNSW index up to date with the `_data` shadow
// table after a fresh connect (or a rollback). Adds rows present in `_data`
// but missing from the graph; if the graph still holds stale keys (deletes
// that were never persisted), rebuilds it from scratch.
// ---------------------------------------------------------------------------

pub(crate) fn reconcile_index(
    db: &VTabConnection,
    config: &VectorTableConfig,
    index: HnswIndex,
    ids_vectors_sql: &str,
) -> Result<HnswIndex> {
    let mut stmt = db.prepare(ids_vectors_sql)?;
    stmt.query(())?;
    let mut count: usize = 0;
    // Streaming: add missing keys incrementally without buffering the whole
    // table. Only on the rare rebuild-from-scratch path below (count
    // mismatch) do we pay for a second scan.
    while let Some(row) = stmt.next()? {
        count += 1;
        let id = row[0].get_i64();
        if !index.contains(id as u64) {
            let vector = row[1].get_blob()?;
            index.add(id as u64, vector).map_err(|e| {
                Error::Module(format!(
                    "reconcile of vector table '{}' failed: {e}",
                    config.table_name
                ))
            })?;
        }
    }
    if index.len() == count {
        return Ok(index);
    }
    // Graph holds keys that no longer exist in _data (deletes lost since the
    // last persist): rebuild from scratch by re-querying `_data`.
    let fresh = config
        .new_index()
        .map_err(|e| Error::Module(e.to_string()))?;
    let mut stmt = db.prepare(ids_vectors_sql)?;
    stmt.query(())?;
    while let Some(row) = stmt.next()? {
        let id = row[0].get_i64();
        let vector = row[1].get_blob()?;
        fresh.add(id as u64, vector).map_err(|e| {
            Error::Module(format!(
                "reconcile of vector table '{}' failed: {e}",
                config.table_name
            ))
        })?;
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
    if verify_against_meta && let Some(meta) = load_meta_from_shadow(db, &config.table_name)? {
        let (dim, vtype, metric, params) =
            VectorTableConfig::params_from_meta(&meta).map_err(|e| Error::Module(e.to_string()))?;
        if dim != config.dim || vtype != config.vtype || metric != config.metric {
            return Err(Error::Module(format!(
                "declared parameters disagree with persisted meta for {}",
                config.table_name
            )));
        }
        if let Some(mode_str) = meta["mode"].as_str() {
            let persisted_mode = crate::vtab::config::IndexMode::from_name(mode_str)
                .map_err(|e| Error::Module(e.to_string()))?;
            if persisted_mode != config.mode {
                return Err(Error::Module(format!(
                    "declared mode disagrees with persisted meta for {}: expected {}, got {}",
                    config.table_name,
                    persisted_mode.name(),
                    config.mode.name()
                )));
            }
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
    // Built before reconcile so its `ids_vectors` SQL can be reused there
    // instead of re-deriving the same string (see TableSql::ids_vectors).
    let sql = TableSql::new(&config);

    let state = if config.mode == crate::vtab::config::IndexMode::Exact {
        // Exact mode has no HNSW index to load/reconcile/snapshot.
        Arc::new(RefCell::new(IndexState {
            index: None,
            dirty: false,
            last_committed: None,
            changes_since_persist: 0,
            destructive_since_persist: false,
        }))
    } else {
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

        let index = reconcile_index(db, &config, index, &sql.ids_vectors)?;

        let snapshot = index
            .save_to_buffer()
            .map_err(|e| Error::Module(e.to_string()))?;
        Arc::new(RefCell::new(IndexState {
            index: Some(index),
            dirty: false,
            last_committed: Some(snapshot),
            changes_since_persist: 0,
            destructive_since_persist: false,
        }))
    };

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
        sql,
        state,
        db: db as *const VTabConnection,
        functions,
        registry: aux.clone(),
        stmt_insert: RefCell::new(None),
        stmt_insert_with_id: RefCell::new(None),
        stmt_delete: RefCell::new(None),
        stmt_fetch_by_id: RefCell::new(None),
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
        let n_meta = self.config.metadata_columns.len() as i32;
        let distance_col = 2 + n_meta;

        // Pass 1: classify. Metadata-column constraints with a supported op
        // become pushable filters instead of forcing has_other.
        let mut has_knn = false;
        let mut knn_count = 0u32;
        let mut has_limit = false;
        let mut has_other = false;
        let mut filters: Vec<(i32, &'static str)> = Vec::new();
        for c in info.constraints() {
            if !c.usable() {
                continue;
            }
            if c.column() == distance_col && matches!(c.op(), ConstraintOp::Function(_)) {
                has_knn = true;
                knn_count += 1;
            } else if matches!(c.op(), ConstraintOp::Limit) {
                has_limit = true;
            } else if matches!(c.op(), ConstraintOp::Offset) {
                // Offset is neither pushed nor blocking on its own.
            } else if c.column() >= 2 && c.column() < 2 + n_meta {
                let op = op_token(c.op());
                match op {
                    Some(op) => filters.push((c.column(), op)),
                    None => has_other = true,
                }
            } else {
                has_other = true;
            }
        }

        if knn_count > 1 {
            return Err(Error::Module(
                "at most one knn_match constraint is supported per query".to_string(),
            ));
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

        // Pass 2: assign argv slots. The wire contract (index_str + argv) is
        // fixed as query, then k (if taken), then filter values in `filters`
        // order — independent of whatever order SQLite hands back constraints
        // from `info.constraints()`. So assign in three separate passes rather
        // than relying on iteration order: knn function constraint first,
        // then limit, then each pushed filter in `filters` order. Filters are
        // left un-omitted so SQLite double-checks them.
        //
        // Each constraint may only be assigned an argv slot ONCE. When two
        // constraints share the same (column, op) — e.g. `score > 1.0 AND
        // score > 3.0` — a naive rescan-from-start would match the *first*
        // one for every filters-list entry, leaving a later constraint with
        // no argv_index and creating a gap that SQLite rejects as an
        // "xBestIndex malfunction". `IndexInfoConstraintIterator` always
        // yields constraints in the same stable position order across calls,
        // so a position-index set is enough to make argv assignment
        // one-to-one with constraints.
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut argv_next: u32 = 1;
        for (pos, mut c) in info.constraints().enumerate() {
            if c.usable()
                && c.column() == distance_col
                && matches!(c.op(), ConstraintOp::Function(_))
            {
                c.set_argv_index(Some(argv_next - 1));
                c.set_omit(true);
                argv_next += 1;
                used.insert(pos);
            }
        }
        if take_limit {
            for (pos, mut c) in info.constraints().enumerate() {
                if c.usable() && matches!(c.op(), ConstraintOp::Limit) {
                    c.set_argv_index(Some(argv_next - 1));
                    argv_next += 1;
                    used.insert(pos);
                }
            }
        }
        for (filter_col, filter_op) in &filters {
            for (pos, mut c) in info.constraints().enumerate() {
                if used.contains(&pos) || !c.usable() || c.column() != *filter_col {
                    continue;
                }
                let Some(op) = op_token(c.op()) else {
                    continue;
                };
                if op == *filter_op {
                    c.set_argv_index(Some(argv_next - 1));
                    argv_next += 1;
                    used.insert(pos);
                    break;
                }
            }
        }

        if has_knn {
            info.set_index_num(INDEX_KNN);
            if has_order_by && order_consumable {
                info.set_order_by_consumed(true);
            }
            info.set_estimated_cost(10.0);
            info.set_estimated_rows(10);
            let spec = format!(
                "knn;limit={};f={}",
                if take_limit { 1 } else { 0 },
                filters
                    .iter()
                    .map(|(c, o)| format!("{c}:{o}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            info.set_index_str(Some(&spec))?;
        } else {
            info.set_index_num(INDEX_SCAN);
            info.set_estimated_cost(1_000_000.0);
            info.set_estimated_rows(1_000_000);
        }

        Ok(())
    }

    fn open(&'vtab self) -> Result<Self::Cursor> {
        Ok(VectorCursor {
            // Placeholder — filter() always runs before any row is read and
            // replaces this with the real Scan or Knn mode.
            mode: CursorMode::Knn {
                results: Vec::new(),
                pos: 0,
            },
            num_metadata_cols: self.config.metadata_columns.len(),
            db: self.db,
            config: &self.config as *const VectorTableConfig,
            sql: &self.sql as *const TableSql,
            state: Arc::clone(&self.state),
        })
    }

    fn disconnect(self) -> DisconnectResult<Self> {
        // Mirror of connect()/build_vtab's `aux.register(...)`: this is the
        // normal (non-DROP) teardown path — closing the connection, or
        // SQLite reloading the schema — and must remove the same registry
        // entry so it doesn't linger as a dead Weak.
        self.registry
            .unregister(&self.config.db_name, &self.config.table_name);
        Ok(())
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
        self.registry
            .unregister(&self.config.db_name, &self.config.table_name);
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
                {
                    let mut stmt = cached_stmt(&self.stmt_delete, db, &self.sql.delete)?;
                    delete_from_data_shadow(stmt.as_mut().unwrap(), rowid)?;
                }
                if let Some(idx) = &self.state.borrow().index {
                    idx.remove(rowid as u64)
                        .map_err(|e| Error::Module(e.to_string()))?;
                }
                {
                    let mut s = self.state.borrow_mut();
                    if s.index.is_some() {
                        s.dirty = true;
                        s.changes_since_persist += 1;
                        // Deletes remove an existing graph key; reconcile-on-
                        // connect only re-adds keys that are *missing*, so an
                        // unpersisted delete must be forced out eagerly (see
                        // IndexState::destructive_since_persist).
                        s.destructive_since_persist = true;
                    }
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

                // Validate dimension and finiteness before inserting.
                // `validate_finite` checks the blob size first, so it covers
                // the dimension check too.
                self.config
                    .vtype
                    .validate_finite(&vector_blob, self.config.dim)
                    .map_err(|e| Error::Module(e.to_string()))?;

                let rowid = if explicit_id.is_some() {
                    let mut stmt =
                        cached_stmt(&self.stmt_insert_with_id, db, &self.sql.insert_with_id)?;
                    insert_into_data_shadow(
                        stmt.as_mut().unwrap(),
                        explicit_id,
                        &vector_blob,
                        meta_args,
                    )?
                } else {
                    let mut stmt = cached_stmt(&self.stmt_insert, db, &self.sql.insert)?;
                    insert_into_data_shadow(
                        stmt.as_mut().unwrap(),
                        explicit_id,
                        &vector_blob,
                        meta_args,
                    )?
                };

                let state = self.state.borrow();
                if let Some(idx) = &state.index {
                    idx.add(rowid as u64, &vector_blob)
                        .map_err(|e| Error::Module(e.to_string()))?;
                }
                drop(state);
                {
                    let mut s = self.state.borrow_mut();
                    if s.index.is_some() {
                        s.dirty = true;
                        s.changes_since_persist += 1;
                    }
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
                // sqlite3_value_nochange). Empirically, this vtab's cursor never
                // opts into the nochange optimization, so
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
                    return Err(Error::Module("vector column cannot be NULL".to_string()));
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
                // Exact mode has no in-memory index to re-key, so skip the
                // reindex machinery (and the extra shadow read below) entirely.
                let has_index = self.state.borrow().index.is_some();
                let needs_reindex = has_index && (rowid_changed || include_vector);

                // If we need to re-key the index but the vector itself isn't
                // changing, fetch the existing vector BEFORE mutating the shadow
                // row (once the row's id changes, it's no longer reachable at
                // old_rowid).
                let reindex_vector: Option<Vec<u8>> = if needs_reindex && vector_blob.is_none() {
                    let mut stmt = cached_stmt(&self.stmt_fetch_by_id, db, &self.sql.fetch_by_id)?;
                    match fetch_row_from_shadow(stmt.as_mut().unwrap(), old_rowid)? {
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
                    let state = self.state.borrow();
                    if let Some(idx) = &state.index {
                        let final_vector = vector_blob
                            .or(reindex_vector)
                            .expect("vector available for reindex: computed above");
                        idx.remove(old_rowid as u64)
                            .map_err(|e| Error::Module(e.to_string()))?;
                        idx.add(new_rowid as u64, &final_vector)
                            .map_err(|e| Error::Module(e.to_string()))?;
                    }
                    drop(state);
                    {
                        let mut s = self.state.borrow_mut();
                        if s.index.is_some() {
                            s.dirty = true;
                            s.changes_since_persist += 1;
                            // An UPDATE re-keys/re-embeds an existing graph
                            // entry in place; `index.contains(id)` stays true
                            // so reconcile-on-connect would skip it and serve
                            // the stale embedding. Force an eager persist.
                            s.destructive_since_persist = true;
                        }
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
            ids_vectors_sql: self.sql.ids_vectors.clone(),
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
