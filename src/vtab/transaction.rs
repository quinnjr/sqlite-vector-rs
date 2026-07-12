use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::Connection;
use sqlite3_ext::query::ToParam;
use sqlite3_ext::vtab::VTabConnection;
use sqlite3_ext::{Error, FromValue, Result};

use crate::index::HnswIndex;
use crate::vtab::config::VectorTableConfig;
use crate::vtab::shadow::ShadowOps;

pub struct IndexState {
    /// `None` iff the table uses `mode=exact` (no HNSW index at all).
    pub index: Option<HnswIndex>,
    pub dirty: bool,
    pub last_committed: Option<Vec<u8>>,
    pub changes_since_persist: u64,
    /// Set by Update/Delete (never by plain Insert — see Finding 1 in the
    /// post-review fix wave): tracks whether a destructive change to an
    /// *existing* graph key happened since the last persist. Reconcile-on-
    /// connect only detects rows missing from the graph (`!index.contains`)
    /// or a `len() != count` mismatch; it cannot detect a key that's present
    /// but stale (updated in place, or deleted-then-reinserted with the same
    /// id and a different vector). Forcing an eager persist on any
    /// destructive op sidesteps that blind spot without giving up the
    /// insert-side batching that `sync_every` exists for.
    pub destructive_since_persist: bool,
}

pub struct VectorTransaction {
    pub state: Arc<RefCell<IndexState>>,
    pub table_name: String,
    /// Safety: valid for the vtab lifetime — SQLite keeps the connection alive.
    pub db: *const VTabConnection,
    pub snapshots: Vec<(i32, Vec<u8>)>,
    pub sync_every: u64,
    pub config: VectorTableConfig,
}

// Safety: VectorTransaction is only ever accessed from a single thread by SQLite.
unsafe impl Send for VectorTransaction {}
unsafe impl Sync for VectorTransaction {}

/// Serialize the index, upsert `hnsw_graph` + `graph_state` rows in the
/// `_index` shadow table, and update the in-memory persistence bookkeeping.
/// Shared by the transaction `sync()` path and the `vector_sync_index` scalar
/// function (which supplies `ctx.db()`, a `&Connection`, rather than the
/// `&VTabConnection` the vtab path has — both deref to the same `Connection`
/// in sqlite3_ext 0.2).
pub fn persist_index(db: &Connection, table_name: &str, s: &mut IndexState) -> Result<()> {
    use sqlite3_ext::query::Statement;
    let Some(index) = &s.index else {
        // Exact mode: no HNSW graph to persist.
        return Ok(());
    };
    let buf = index
        .save_to_buffer()
        .map_err(|e| Error::Module(e.to_string()))?;
    let sql = ShadowOps::upsert_index_sql(table_name);
    db.insert(&sql, |stmt: &mut Statement| {
        "hnsw_graph".bind_param(&mut *stmt, 1)?;
        buf.as_slice().bind_param(&mut *stmt, 2)?;
        Ok(())
    })?;

    let (count, max_id): (i64, i64) = db.query_row(
        &format!("SELECT count(*), coalesce(max(id), 0) FROM \"{table_name}_data\""),
        (),
        |row| Ok((row[0].get_i64(), row[1].get_i64())),
    )?;
    let state_json = format!("{{\"row_count\": {count}, \"max_rowid\": {max_id}}}");
    db.insert(&sql, |stmt: &mut Statement| {
        "graph_state".bind_param(&mut *stmt, 1)?;
        state_json.as_str().bind_param(&mut *stmt, 2)?;
        Ok(())
    })?;

    s.last_committed = Some(buf);
    s.changes_since_persist = 0;
    s.destructive_since_persist = false;
    s.dirty = false;
    Ok(())
}

impl sqlite3_ext::vtab::VTabTransaction for VectorTransaction {
    fn sync(&mut self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if s.dirty && (s.destructive_since_persist || s.changes_since_persist >= self.sync_every) {
            let db = unsafe { &*self.db };
            persist_index(db, &self.table_name, &mut s)?;
        }
        Ok(())
    }

    fn commit(self) -> Result<()> {
        // sync() has already serialized and persisted, if the threshold was hit;
        // otherwise the committed-but-unpersisted rows survive via reconcile on
        // the next connect (or a subsequent rollback, see `rollback` below).
        Ok(())
    }

    fn rollback(self) -> Result<()> {
        if self.state.borrow().index.is_none() {
            // Exact mode: no HNSW graph to roll back.
            return Ok(());
        }
        let db = unsafe { &*self.db };
        {
            let mut s = self.state.borrow_mut();
            let buf = s
                .last_committed
                .clone()
                .expect("last_committed primed at connect/create");
            s.index
                .as_ref()
                .expect("checked above")
                .load_from_buffer(&buf)
                .map_err(|e| Error::Module(e.to_string()))?;
            s.dirty = false;
            // The restored snapshot equals `last_committed`, so there are no
            // pending (unpersisted) changes yet. The reconcile pass below may
            // re-add rows that are already in `_data` but not yet in the
            // graph; those additions are themselves unpersisted, but
            // `reconcile_index` doesn't currently report a count, so we
            // conservatively reset to 0 here and let the existing
            // `sync_every`/dirty-tracking on subsequent mutations (or a
            // future explicit `vector_sync_index`) catch up. This matches
            // the "reset to 0" fallback called out in the review finding.
            s.changes_since_persist = 0;
            s.destructive_since_persist = false;
        }
        // `last_committed` may predate commits that survived this rollback
        // (rows persisted to `_data` before the rollback but never persisted
        // to the graph): replay them from `_data`, same code path as connect.
        //
        // IMPORTANT: build the reconciled index from a *moved-out* snapshot
        // index without mutating `state` first. If `reconcile_index` errors,
        // put the original (snapshot-restored) index back into `state`
        // before propagating the error, so the table never serves from an
        // empty placeholder index.
        let mut s = self.state.borrow_mut();
        let placeholder = HnswIndex::new(
            self.config.dim,
            self.config.vtype,
            self.config.metric,
            Some(self.config.hnsw_params),
        )
        .map_err(|e| Error::Module(e.to_string()))?;
        let index = s
            .index
            .replace(placeholder)
            .expect("checked Some at top of rollback");
        match crate::vtab::reconcile_index(db, &self.config, index) {
            Ok(reconciled) => {
                s.index = Some(reconciled);
                Ok(())
            }
            Err(e) => {
                // `index` was moved into `reconcile_index`; on error we no
                // longer have it back, so rebuild the same snapshot-restored
                // state from `last_committed` (which is exactly what `index`
                // held before the reconcile attempt) rather than leaving the
                // empty placeholder installed.
                let buf = s
                    .last_committed
                    .clone()
                    .expect("last_committed primed at connect/create");
                s.index
                    .as_ref()
                    .expect("checked Some at top of rollback")
                    .load_from_buffer(&buf)
                    .map_err(|load_err| Error::Module(load_err.to_string()))?;
                Err(e)
            }
        }
    }

    fn savepoint(&mut self, n: i32) -> Result<()> {
        let s = self.state.borrow();
        let Some(index) = &s.index else {
            // Exact mode: no HNSW graph to snapshot.
            return Ok(());
        };
        let buf = index
            .save_to_buffer()
            .map_err(|e| Error::Module(e.to_string()))?;
        drop(s);
        self.snapshots.push((n, buf));
        Ok(())
    }

    fn release(&mut self, n: i32) -> Result<()> {
        self.snapshots.retain(|(sp, _)| *sp < n);
        Ok(())
    }

    fn rollback_to(&mut self, n: i32) -> Result<()> {
        let s = self.state.borrow_mut();
        let Some(idx_ref) = &s.index else {
            // Exact mode: no HNSW graph to roll back.
            return Ok(());
        };
        if let Some(idx) = self.snapshots.iter().position(|(sp, _)| *sp >= n) {
            idx_ref
                .load_from_buffer(&self.snapshots[idx].1)
                .map_err(|e| Error::Module(e.to_string()))?;
            self.snapshots.truncate(idx + 1);
        } else if let Some(ref buf) = s.last_committed.clone() {
            idx_ref
                .load_from_buffer(buf)
                .map_err(|e| Error::Module(e.to_string()))?;
        }
        Ok(())
    }
}
