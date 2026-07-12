use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::query::ToParam;
use sqlite3_ext::vtab::VTabConnection;
use sqlite3_ext::{Error, Result};

use crate::index::HnswIndex;
use crate::vtab::shadow::ShadowOps;

pub struct IndexState {
    pub index: HnswIndex,
    pub dirty: bool,
    pub last_committed: Option<Vec<u8>>,
}

pub struct VectorTransaction {
    pub state: Arc<RefCell<IndexState>>,
    pub table_name: String,
    /// Safety: valid for the vtab lifetime — SQLite keeps the connection alive.
    pub db: *const VTabConnection,
    pub snapshots: Vec<(i32, Vec<u8>)>,
}

// Safety: VectorTransaction is only ever accessed from a single thread by SQLite.
unsafe impl Send for VectorTransaction {}
unsafe impl Sync for VectorTransaction {}

impl sqlite3_ext::vtab::VTabTransaction for VectorTransaction {
    fn sync(&mut self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if s.dirty {
            let buf = s
                .index
                .save_to_buffer()
                .map_err(|e| Error::Module(e.to_string()))?;

            // Persist serialized HNSW graph to the _index shadow table.
            use sqlite3_ext::query::Statement;
            let db = unsafe { &*self.db };
            let sql = ShadowOps::upsert_index_sql(&self.table_name);
            db.insert(&sql, |stmt: &mut Statement| {
                "hnsw_graph".bind_param(&mut *stmt, 1)?;
                buf.as_slice().bind_param(&mut *stmt, 2)?;
                Ok(())
            })?;

            s.last_committed = Some(buf);
            s.dirty = false;
        }
        Ok(())
    }

    fn commit(self) -> Result<()> {
        // sync() has already serialized and persisted; nothing more to do.
        Ok(())
    }

    fn rollback(self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if let Some(ref buf) = s.last_committed.clone() {
            s.index
                .load_from_buffer(buf)
                .map_err(|e| Error::Module(e.to_string()))?;
        }
        s.dirty = false;
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
