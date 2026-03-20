use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::{Error, Result};

use crate::index::HnswIndex;

pub struct IndexState {
    pub index: HnswIndex,
    pub dirty: bool,
    pub last_committed: Option<Vec<u8>>,
}

pub struct VectorTransaction {
    pub state: Arc<RefCell<IndexState>>,
    pub table_name: String,
}

impl sqlite3_ext::vtab::VTabTransaction for VectorTransaction {
    fn sync(&mut self) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if s.dirty {
            let buf = s
                .index
                .save_to_buffer()
                .map_err(|e| Error::Module(e.to_string()))?;
            s.last_committed = Some(buf);
            s.dirty = false;
        }
        Ok(())
    }

    fn commit(self) -> Result<()> {
        // sync() has already serialized; nothing more to do.
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

    fn savepoint(&mut self, _n: i32) -> Result<()> {
        Ok(())
    }

    fn release(&mut self, _n: i32) -> Result<()> {
        Ok(())
    }

    fn rollback_to(&mut self, _n: i32) -> Result<()> {
        let mut s = self.state.borrow_mut();
        if let Some(ref buf) = s.last_committed.clone() {
            s.index
                .load_from_buffer(buf)
                .map_err(|e| Error::Module(e.to_string()))?;
        }
        s.dirty = false;
        Ok(())
    }
}
