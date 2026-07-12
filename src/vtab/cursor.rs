use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::{
    Error, FallibleIteratorMut, FromValue, Result, Value, ValueRef,
    query::QueryResult,
    vtab::{ColumnContext, VTabConnection, VTabCursor},
};

use crate::vtab::TableSql;
use crate::vtab::config::VectorTableConfig;
use crate::vtab::transaction::IndexState;

// Index number must match INDEX_KNN in mod.rs
const INDEX_KNN: i32 = 1;

pub enum CursorMode {
    Scan {
        stmt: sqlite3_ext::query::Statement,
        current: Option<ScanRow>,
    },
    Knn {
        results: Vec<KnnRow>,
        pos: usize,
    },
}

pub struct ScanRow {
    pub id: i64,
    pub vector: Vec<u8>,
    pub metadata: Vec<Value>,
}

pub struct KnnRow {
    pub id: i64,
    pub vector: Vec<u8>,
    pub metadata: Vec<Value>,
    pub distance: f64,
}

pub struct VectorCursor {
    pub mode: CursorMode,
    pub num_metadata_cols: usize,
    /// Safety: valid for the vtab lifetime — SQLite keeps the connection alive.
    pub db: *const VTabConnection,
    /// Safety: valid for the vtab lifetime — VectorTable owns the config.
    pub config: *const VectorTableConfig,
    /// Safety: valid for the vtab lifetime — VectorTable owns the prebuilt SQL.
    pub sql: *const TableSql,
    pub state: Arc<RefCell<IndexState>>,
}

// Safety: VectorCursor is only ever accessed from a single thread by SQLite.
// This covers the raw db/config pointers and the owned `Statement` inside
// `CursorMode::Scan` (which wraps a live `sqlite3_stmt*`): all are created and
// used exclusively under SQLite's one-thread-per-connection guarantee.
unsafe impl Send for VectorCursor {}
unsafe impl Sync for VectorCursor {}

impl VectorCursor {
    fn current_id(&self) -> i64 {
        match &self.mode {
            CursorMode::Scan { current, .. } => current.as_ref().expect("eof checked").id,
            CursorMode::Knn { results, pos } => results[*pos].id,
        }
    }

    fn current_vector(&self) -> &[u8] {
        match &self.mode {
            CursorMode::Scan { current, .. } => &current.as_ref().expect("eof checked").vector,
            CursorMode::Knn { results, pos } => &results[*pos].vector,
        }
    }

    fn current_metadata(&self) -> &[Value] {
        match &self.mode {
            CursorMode::Scan { current, .. } => &current.as_ref().expect("eof checked").metadata,
            CursorMode::Knn { results, pos } => &results[*pos].metadata,
        }
    }

    fn current_distance(&self) -> Option<f64> {
        match &self.mode {
            CursorMode::Scan { .. } => None,
            CursorMode::Knn { results, pos } => Some(results[*pos].distance),
        }
    }
}

impl VTabCursor for VectorCursor {
    fn filter(
        &mut self,
        index_num: i32,
        _index_str: Option<&str>,
        args: &mut [&mut ValueRef],
    ) -> Result<()> {
        // Safety: db, config, and sql pointers are valid for the vtab lifetime.
        let db = unsafe { &*self.db };
        let config = unsafe { &*self.config };
        let sql = unsafe { &*self.sql };

        match index_num {
            INDEX_KNN => {
                // args[0] = query vector blob (from knn_match function constraint)
                // args[1] = k (from LIMIT clause, if present)
                if args.is_empty() {
                    return Err(Error::Module(
                        "knn_match requires a query vector argument".into(),
                    ));
                }
                let query_blob = args[0].get_blob()?.to_vec();
                let k = if args.len() > 1 {
                    args[1].get_i64() as usize
                } else {
                    // Default k when no LIMIT is specified
                    crate::vtab::DEFAULT_KNN_K
                };

                let state = self.state.borrow();
                let hits = state
                    .index
                    .search(&query_blob, k)
                    .map_err(|e| Error::Module(e.to_string()))?;

                let num_meta = config.metadata_columns.len();
                let mut stmt = db.prepare(&sql.fetch_by_id)?;
                let mut results = Vec::with_capacity(hits.len());
                for (key, dist) in hits {
                    stmt.query([key as i64])?;
                    if let Some(row) = stmt.next()? {
                        let r = read_scan_row(row, num_meta)?;
                        results.push(KnnRow {
                            id: r.id,
                            vector: r.vector,
                            metadata: r.metadata,
                            distance: dist as f64,
                        });
                    }
                }
                self.mode = CursorMode::Knn { results, pos: 0 };
            }
            _ => {
                let mut stmt = db.prepare(&sql.scan_all)?;
                stmt.query(())?;
                let mut mode = CursorMode::Scan {
                    stmt,
                    current: None,
                };
                advance_scan(&mut mode, config.metadata_columns.len())?;
                self.mode = mode;
            }
        }

        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        match &mut self.mode {
            CursorMode::Scan { .. } => {
                advance_scan(&mut self.mode, self.num_metadata_cols)?;
            }
            CursorMode::Knn { pos, .. } => {
                *pos += 1;
            }
        }
        Ok(())
    }

    fn eof(&mut self) -> bool {
        match &self.mode {
            CursorMode::Scan { current, .. } => current.is_none(),
            CursorMode::Knn { results, pos } => *pos >= results.len(),
        }
    }

    fn column(&mut self, idx: usize, ctx: &ColumnContext) -> Result<()> {
        // Column layout: 0=id, 1=vector, 2..2+N=metadata[0..N], last=distance
        match idx {
            0 => {
                ctx.set_result(self.current_id())?;
            }
            1 => {
                ctx.set_result(self.current_vector())?;
            }
            i if i >= 2 && i < 2 + self.num_metadata_cols => {
                ctx.set_result(self.current_metadata()[i - 2].clone())?;
            }
            _ => {
                // distance column (last)
                match self.current_distance() {
                    Some(d) => ctx.set_result(d)?,
                    None => ctx.set_result(())?,
                }
            }
        }
        Ok(())
    }

    fn rowid(&mut self) -> Result<i64> {
        Ok(self.current_id())
    }
}

// ---------------------------------------------------------------------------
// Helpers duplicated here to avoid circular imports (mirror mod.rs helpers)
// ---------------------------------------------------------------------------

/// Read a single row out of a live `_data` query result into an owned `ScanRow`.
/// Shared by the streaming full scan and KNN's per-id lookup.
fn read_scan_row(row: &mut QueryResult, num_meta: usize) -> Result<ScanRow> {
    let id = row[0].get_i64();
    let vector = row[1].get_blob()?.to_vec();
    let mut metadata = Vec::with_capacity(num_meta);
    for i in 0..num_meta {
        metadata.push(row[2 + i].to_owned()?);
    }
    Ok(ScanRow {
        id,
        vector,
        metadata,
    })
}

/// Advance a `CursorMode::Scan` by one row, reading directly from the live
/// statement instead of a pre-materialized buffer. No-op for KNN mode.
fn advance_scan(mode: &mut CursorMode, num_meta: usize) -> Result<()> {
    if let CursorMode::Scan { stmt, current } = mode {
        *current = match stmt.next()? {
            Some(row) => Some(read_scan_row(row, num_meta)?),
            None => None,
        };
    }
    Ok(())
}
