use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::{
    Error, FallibleIteratorMut, FromValue, Result, Value, ValueRef,
    query::{QueryResult, ToParam},
    vtab::{ColumnContext, VTabConnection, VTabCursor},
};

use crate::vtab::TableSql;
use crate::vtab::config::{IndexMode, VectorTableConfig};
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
        index_str: Option<&str>,
        args: &mut [&mut ValueRef],
    ) -> Result<()> {
        // Safety: db, config, and sql pointers are valid for the vtab lifetime.
        let db = unsafe { &*self.db };
        let config = unsafe { &*self.config };
        let sql = unsafe { &*self.sql };

        match index_num {
            INDEX_KNN => {
                // args[0] = query vector blob (from knn_match function constraint)
                // args[1] = k (from LIMIT clause, if present, per index_str's limit=1)
                // args[1 or 2..] = pushed metadata filter values, in index_str's f= order
                if args.is_empty() {
                    return Err(Error::Module(
                        "knn_match requires a query vector argument".into(),
                    ));
                }
                let query_blob = args[0].get_blob()?.to_vec();

                let spec = index_str.unwrap_or("knn;limit=0;f=");
                let limit_taken = spec.contains("limit=1");
                let filter_part = spec.rsplit("f=").next().unwrap_or("");
                let filters: Vec<(usize, &str)> = filter_part
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let (col, op) = s.split_once(':').expect("index_str built by best_index");
                        (col.parse::<usize>().unwrap(), op)
                    })
                    .collect();

                let mut next_arg = 1;
                let k = if limit_taken {
                    let k = args[next_arg].get_i64() as usize;
                    next_arg += 1;
                    k
                } else {
                    // Default k when no LIMIT is specified
                    crate::vtab::DEFAULT_KNN_K
                };
                let filter_args = &mut args[next_arg..];

                let num_meta = config.metadata_columns.len();
                let mut fetch_sql = sql.fetch_by_id.clone();
                for (col, op) in &filters {
                    let name = &config.metadata_columns[col - 2].0;
                    let sql_op = match *op {
                        "eq" => "=",
                        "gt" => ">",
                        "ge" => ">=",
                        "lt" => "<",
                        "le" => "<=",
                        _ => unreachable!("index_str built by best_index"),
                    };
                    fetch_sql.push_str(&format!(" AND {name} {sql_op} ?"));
                }

                let target_k = k;
                // Short-circuit: when target_k == 0, return no rows immediately
                if target_k == 0 {
                    self.mode = CursorMode::Knn { results: Vec::new(), pos: 0 };
                    return Ok(());
                }

                if config.mode == IndexMode::Exact {
                    // Brute-force streaming top-k with pushed filters applied in SQL.
                    let mut scan_sql = format!("{} WHERE 1=1", sql.scan_all);
                    for (col, op) in &filters {
                        let name = &config.metadata_columns[col - 2].0;
                        let sql_op = match *op {
                            "eq" => "=",
                            "gt" => ">",
                            "ge" => ">=",
                            "lt" => "<",
                            "le" => "<=",
                            _ => unreachable!("index_str built by best_index"),
                        };
                        scan_sql.push_str(&format!(" AND {name} {sql_op} ?"));
                    }
                    let mut stmt = db.prepare(&scan_sql)?;
                    stmt.query(|q: &mut sqlite3_ext::query::Statement| {
                        for (i, arg) in filter_args.iter_mut().enumerate() {
                            let vref: &ValueRef = arg;
                            vref.bind_param(q, (i + 1) as i32)?;
                        }
                        Ok(())
                    })?;

                    // Max-heap of size k ordered by distance (f64::total_cmp), so
                    // the worst-of-the-best-k-so-far is always at the top and can
                    // be evicted in O(log k) as better candidates stream in.
                    struct Hit(f64, ScanRow);
                    impl PartialEq for Hit {
                        fn eq(&self, o: &Self) -> bool {
                            self.0 == o.0
                        }
                    }
                    impl Eq for Hit {}
                    impl PartialOrd for Hit {
                        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                            Some(self.cmp(o))
                        }
                    }
                    impl Ord for Hit {
                        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                            self.0.total_cmp(&o.0)
                        }
                    }

                    let mut heap: std::collections::BinaryHeap<Hit> =
                        std::collections::BinaryHeap::with_capacity(target_k + 1);
                    while let Some(row) = stmt.next()? {
                        let r = read_scan_row(row, num_meta)?;
                        let d = crate::distance::compute_distance(
                            &query_blob,
                            &r.vector,
                            config.vtype,
                            config.metric,
                            config.dim,
                        )
                        .map_err(|e| Error::Module(e.to_string()))?;
                        heap.push(Hit(d, r));
                        if heap.len() > target_k {
                            heap.pop();
                        }
                    }
                    let mut rows: Vec<Hit> = heap.into_vec();
                    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let results = rows
                        .into_iter()
                        .map(|Hit(d, r)| KnnRow {
                            id: r.id,
                            vector: r.vector,
                            metadata: r.metadata,
                            distance: d,
                        })
                        .collect();
                    self.mode = CursorMode::Knn { results, pos: 0 };
                    return Ok(());
                }

                let index_len = {
                    let state = self.state.borrow();
                    state
                        .index
                        .as_ref()
                        .expect("HNSW index present when mode != Exact")
                        .len()
                };
                let mut kp = (4 * target_k).max(target_k + 16).min(index_len.max(1));
                let mut results: Vec<KnnRow>;
                let mut stmt = db.prepare(&fetch_sql)?;
                loop {
                    let hits = {
                        let state = self.state.borrow();
                        state
                            .index
                            .as_ref()
                            .expect("HNSW index present when mode != Exact")
                            .search(&query_blob, kp)
                            .map_err(|e| Error::Module(e.to_string()))?
                    };
                    results = Vec::with_capacity(target_k);
                    for (key, dist) in &hits {
                        stmt.query(|q: &mut sqlite3_ext::query::Statement| {
                            (*key as i64).bind_param(q, 1)?;
                            for (i, arg) in filter_args.iter_mut().enumerate() {
                                let vref: &ValueRef = arg;
                                vref.bind_param(q, (i + 2) as i32)?;
                            }
                            Ok(())
                        })?;
                        if let Some(row) = stmt.next()? {
                            let r = read_scan_row(row, num_meta)?;
                            results.push(KnnRow {
                                id: r.id,
                                vector: r.vector,
                                metadata: r.metadata,
                                distance: *dist as f64,
                            });
                            if results.len() >= target_k {
                                break;
                            }
                        }
                    }
                    if results.len() >= target_k || kp >= index_len {
                        break;
                    }
                    kp = (kp * 2).min(index_len);
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
