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
    pub index: HnswIndex,
    pub dirty: bool,
    pub last_committed: Option<Vec<u8>>,
    pub changes_since_persist: u64,
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
    let buf = s
        .index
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
    s.dirty = false;
    Ok(())
}

impl sqlite3_ext::vtab::VTabTransaction for VectorTransaction {
    fn sync(&mut self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if s.dirty && s.changes_since_persist >= self.sync_every {
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
        let db = unsafe { &*self.db };
        {
            let mut s = self.state.borrow_mut();
            let buf = s
                .last_committed
                .clone()
                .expect("last_committed primed at connect/create");
            s.index
                .load_from_buffer(&buf)
                .map_err(|e| Error::Module(e.to_string()))?;
            s.dirty = false;
        }
        // `last_committed` may predate commits that survived this rollback
        // (rows persisted to `_data` before the rollback but never persisted
        // to the graph): replay them from `_data`, same code path as connect.
        let mut s = self.state.borrow_mut();
        let placeholder = HnswIndex::new(
            self.config.dim,
            self.config.vtype,
            self.config.metric,
            Some(self.config.hnsw_params),
        )
        .map_err(|e| Error::Module(e.to_string()))?;
        let index = std::mem::replace(&mut s.index, placeholder);
        s.index = crate::vtab::reconcile_index(db, &self.config, index)?;
        Ok(())
    }

    fn savepoint(&mut self, n: i32) -> Result<()> {
        let buf = self
            .state
            .borrow()
            .index
            .save_to_buffer()
            .map_err(|e| Error::Module(e.to_string()))?;
        self.snapshots.push((n, buf));
        Ok(())
    }

    fn release(&mut self, n: i32) -> Result<()> {
        self.snapshots.retain(|(sp, _)| *sp < n);
        Ok(())
    }

    fn rollback_to(&mut self, n: i32) -> Result<()> {
        let s = self.state.borrow_mut();
        if let Some(idx) = self.snapshots.iter().position(|(sp, _)| *sp >= n) {
            s.index
                .load_from_buffer(&self.snapshots[idx].1)
                .map_err(|e| Error::Module(e.to_string()))?;
            self.snapshots.truncate(idx + 1);
        } else if let Some(ref buf) = s.last_committed.clone() {
            s.index
                .load_from_buffer(buf)
                .map_err(|e| Error::Module(e.to_string()))?;
        }
        Ok(())
    }
}
