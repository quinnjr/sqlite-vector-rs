use sqlite3_ext::{
    vtab::{ColumnContext, VTabCursor},
    Result, ValueRef,
};

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
}

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
        _index_num: i32,
        _index_str: Option<&str>,
        _args: &mut [&mut ValueRef],
    ) -> Result<()> {
        self.set_pos(0);
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
