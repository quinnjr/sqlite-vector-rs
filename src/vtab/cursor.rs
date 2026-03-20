use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::{
    vtab::{ColumnContext, VTabCursor, VTabConnection},
    Error, FallibleIteratorMut, FromValue, Result, ValueRef,
};

use crate::vtab::config::VectorTableConfig;
use crate::vtab::shadow::ShadowOps;
use crate::vtab::transaction::IndexState;

// Index numbers must match those in mod.rs
const INDEX_SCAN: i32 = 0;
const INDEX_KNN: i32 = 1;

pub enum CursorMode {
    Scan { rows: Vec<ScanRow>, pos: usize },
    Knn { results: Vec<KnnRow>, pos: usize },
}

pub struct ScanRow {
    pub id: i64,
    pub vector: Vec<u8>,
    pub metadata: Vec<Option<Vec<u8>>>,
}

pub struct KnnRow {
    pub id: i64,
    pub vector: Vec<u8>,
    pub metadata: Vec<Option<Vec<u8>>>,
    pub distance: f64,
}

pub struct VectorCursor {
    pub mode: CursorMode,
    pub num_metadata_cols: usize,
    /// Safety: valid for the vtab lifetime — SQLite keeps the connection alive.
    pub db: *const VTabConnection,
    /// Safety: valid for the vtab lifetime — VectorTable owns the config.
    pub config: *const VectorTableConfig,
    pub state: Arc<RefCell<IndexState>>,
}

// Safety: VectorCursor is only ever accessed from a single thread by SQLite.
unsafe impl Send for VectorCursor {}
unsafe impl Sync for VectorCursor {}

impl VectorCursor {
    fn current_id(&self) -> i64 {
        match &self.mode {
            CursorMode::Scan { rows, pos } => rows[*pos].id,
            CursorMode::Knn { results, pos } => results[*pos].id,
        }
    }

    fn current_vector(&self) -> &[u8] {
        match &self.mode {
            CursorMode::Scan { rows, pos } => &rows[*pos].vector,
            CursorMode::Knn { results, pos } => &results[*pos].vector,
        }
    }

    fn current_metadata(&self) -> &[Option<Vec<u8>>] {
        match &self.mode {
            CursorMode::Scan { rows, pos } => &rows[*pos].metadata,
            CursorMode::Knn { results, pos } => &results[*pos].metadata,
        }
    }

    fn current_distance(&self) -> Option<f64> {
        match &self.mode {
            CursorMode::Scan { .. } => None,
            CursorMode::Knn { results, pos } => Some(results[*pos].distance),
        }
    }

    fn len(&self) -> usize {
        match &self.mode {
            CursorMode::Scan { rows, .. } => rows.len(),
            CursorMode::Knn { results, .. } => results.len(),
        }
    }

    fn pos(&self) -> usize {
        match &self.mode {
            CursorMode::Scan { pos, .. } => *pos,
            CursorMode::Knn { pos, .. } => *pos,
        }
    }

    fn set_pos(&mut self, new_pos: usize) {
        match &mut self.mode {
            CursorMode::Scan { pos, .. } => *pos = new_pos,
            CursorMode::Knn { pos, .. } => *pos = new_pos,
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
        // Safety: db and config pointers are valid for the vtab lifetime.
        let db = unsafe { &*self.db };
        let config = unsafe { &*self.config };

        match index_num {
            INDEX_KNN => {
                // args[0] encodes the query blob (from knn_match constraint)
                // args[0] = query vector blob
                // args[1] = k (number of results)
                if args.len() < 2 {
                    return Err(Error::Module(
                        "knn_match requires query vector and k arguments".into(),
                    ));
                }
                let query_blob = args[0].get_blob()?.to_vec();
                let k = args[1].get_i64() as usize;

                let state = self.state.borrow();
                let hits = state
                    .index
                    .search(&query_blob, k)
                    .map_err(|e| Error::Module(e.to_string()))?;

                let mut results = Vec::with_capacity(hits.len());
                for (key, dist) in hits {
                    if let Some(row) = fetch_row_by_id(db, config, key as i64)? {
                        results.push(KnnRow {
                            id: row.id,
                            vector: row.vector,
                            metadata: row.metadata,
                            distance: dist as f64,
                        });
                    }
                }
                self.mode = CursorMode::Knn { results, pos: 0 };
            }
            INDEX_SCAN | _ => {
                let rows = scan_all_rows(db, config)?;
                self.mode = CursorMode::Scan { rows, pos: 0 };
            }
        }

        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        let new_pos = self.pos() + 1;
        self.set_pos(new_pos);
        Ok(())
    }

    fn eof(&mut self) -> bool {
        self.pos() >= self.len()
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
                let meta_idx = i - 2;
                match &self.current_metadata()[meta_idx] {
                    Some(blob) => ctx.set_result(blob.as_slice())?,
                    None => ctx.set_result(())?,
                }
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

fn scan_all_rows(db: &VTabConnection, config: &VectorTableConfig) -> Result<Vec<ScanRow>> {
    let sql = ShadowOps::select_all_data_sql(&config.table_name);
    let num_meta = config.metadata_columns.len();
    let mut stmt = db.prepare(&sql)?;
    stmt.query(())?;
    let mut rows = Vec::new();
    while let Some(row) = stmt.next()? {
        let id = row[0].get_i64();
        let vector = row[1].get_blob()?.to_vec();
        let mut metadata = Vec::with_capacity(num_meta);
        for i in 0..num_meta {
            if row[2 + i].is_null() {
                metadata.push(None);
            } else {
                metadata.push(Some(row[2 + i].get_blob()?.to_vec()));
            }
        }
        rows.push(ScanRow { id, vector, metadata });
    }
    Ok(rows)
}

fn fetch_row_by_id(
    db: &VTabConnection,
    config: &VectorTableConfig,
    id: i64,
) -> Result<Option<ScanRow>> {
    use sqlite3_ext::SQLITE_EMPTY;
    let sql = ShadowOps::select_data_sql(&config.table_name);
    let num_meta = config.metadata_columns.len();
    match db.query_row(&sql, [id], |row| {
        let id = row[0].get_i64();
        let vector = row[1].get_blob()?.to_vec();
        let mut metadata = Vec::with_capacity(num_meta);
        for i in 0..num_meta {
            if row[2 + i].is_null() {
                metadata.push(None);
            } else {
                metadata.push(Some(row[2 + i].get_blob()?.to_vec()));
            }
        }
        Ok(ScanRow { id, vector, metadata })
    }) {
        Ok(row) => Ok(Some(row)),
        Err(ref e) if *e == SQLITE_EMPTY => Ok(None),
        Err(e) => Err(e),
    }
}
