use sqlite3_ext::function::FunctionOptions;
use sqlite3_ext::query::ToParam;
use sqlite3_ext::*;

use crate::arrow_io;
use crate::distance::{DistanceMetric, compute_distance};
use crate::index::HnswIndex;
use crate::json::{blob_to_json, json_to_blob};
use crate::types::VectorType;
use crate::vtab::Registry;
use crate::vtab::shadow::ShadowOps;
use crate::vtab::transaction::persist_index;

/// Register all standalone scalar functions on a connection.
pub fn register_scalar_functions(db: &Connection, registry: Registry) -> Result<()> {
    // vector_distance(blob_a, blob_b, metric, type) -> REAL
    db.create_scalar_function(
        "vector_distance",
        &FunctionOptions::default()
            .set_n_args(4)
            .set_deterministic(true),
        |ctx, args| {
            // Collect string args as owned values first to avoid borrow conflicts
            // with the blob borrows that follow.
            let metric_name = args[2].get_str()?.to_owned();
            let type_name = args[3].get_str()?.to_owned();
            let blob_a = args[0].get_blob()?.to_vec();
            let blob_b = args[1].get_blob()?.to_vec();

            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let metric = DistanceMetric::from_name(&metric_name)
                .map_err(|e| Error::Module(e.to_string()))?;

            let dim = blob_a.len() / vtype.element_size();
            let dist = compute_distance(&blob_a, &blob_b, vtype, metric, dim)
                .map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(dist)?;
            Ok(())
        },
    )?;

    // vector_from_json(json_text, type) -> BLOB
    db.create_scalar_function(
        "vector_from_json",
        &FunctionOptions::default()
            .set_n_args(2)
            .set_deterministic(true),
        |ctx, args| {
            let json_text = args[0].get_str()?.to_owned();
            let type_name = args[1].get_str()?.to_owned();

            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let blob = json_to_blob(&json_text, vtype).map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(&blob[..])?;
            Ok(())
        },
    )?;

    // vector_to_json(blob, type) -> TEXT
    db.create_scalar_function(
        "vector_to_json",
        &FunctionOptions::default()
            .set_n_args(2)
            .set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let blob = args[0].get_blob()?.to_vec();

            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let json = blob_to_json(&blob, vtype).map_err(|e| Error::Module(e.to_string()))?;

            // Pass owned String — ToContextResult is implemented for String
            ctx.set_result(json)?;
            Ok(())
        },
    )?;

    // vector_dims(blob, type) -> INTEGER
    db.create_scalar_function(
        "vector_dims",
        &FunctionOptions::default()
            .set_n_args(2)
            .set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let blob = args[0].get_blob()?;

            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let dims = blob.len() / vtype.element_size();

            ctx.set_result(dims as i64)?;
            Ok(())
        },
    )?;

    // knn_match(col, query_blob) — placeholder for xFindFunction override.
    // The global version is a no-op; the vtab's FindFunctionVTab replaces it
    // when the first argument is a virtual table column.
    db.create_scalar_function(
        "knn_match",
        &FunctionOptions::default().set_n_args(2),
        |ctx, _args| {
            ctx.set_result(1i32)?;
            Ok(())
        },
    )?;

    // vector_rebuild_index(table_name) -> INTEGER (row count)
    //
    // Reads the persisted table config from the `meta` row in the `_index`
    // shadow table, reads all vectors from the shadow data table, builds a
    // fresh HNSW index using the table's real dim/type/metric/HNSW params,
    // and serializes it back to the shadow index table. Returns the number
    // of vectors indexed.
    //
    // NOTE: This writes directly to shadow tables, bypassing the vtab's
    // in-memory index. A running vtab won't see the rebuilt index until
    // reconnect. Intended for offline maintenance, not live use.
    db.create_scalar_function(
        "vector_rebuild_index",
        &FunctionOptions::default().set_n_args(1),
        |ctx, args| {
            let table_name = args[0].get_str()?.to_owned();
            let db = ctx.db();

            let meta_json = db.query_row(
                &ShadowOps::select_index_sql(&table_name),
                ["meta"],
                |row| Ok(row[0].get_str()?.to_owned()),
            )?;
            let meta: serde_json::Value =
                serde_json::from_str(&meta_json).map_err(|e| Error::Module(e.to_string()))?;
            let (dim, vtype, metric, params) =
                crate::vtab::config::VectorTableConfig::params_from_meta(&meta)
                    .map_err(|e| Error::Module(e.to_string()))?;

            // Read all (rowid, vector_blob) pairs from the data shadow table.
            let sql = ShadowOps::select_all_data_sql(&table_name);
            let mut stmt = db.prepare(&sql)?;
            stmt.query(())?;

            let mut rows: Vec<(i64, Vec<u8>)> = Vec::new();
            while let Some(row) = stmt.next()? {
                let id = row[0].get_i64();
                let blob = row[1].get_blob()?.to_vec();
                rows.push((id, blob));
            }

            if rows.is_empty() {
                ctx.set_result(0i64)?;
                return Ok(());
            }

            // Build a fresh index (using the persisted params) and insert every vector.
            let index = HnswIndex::new(dim, vtype, metric, Some(params))
                .map_err(|e| Error::Module(e.to_string()))?;
            for (id, blob) in &rows {
                index
                    .add(*id as u64, blob)
                    .map_err(|e| Error::Module(e.to_string()))?;
            }

            // Serialize and persist to the _index shadow table.
            let buf = index
                .save_to_buffer()
                .map_err(|e| Error::Module(e.to_string()))?;
            let upsert_sql = ShadowOps::upsert_index_sql(&table_name);
            db.insert(&upsert_sql, |stmt: &mut query::Statement| {
                "hnsw_graph".bind_param(&mut *stmt, 1)?;
                buf.as_slice().bind_param(&mut *stmt, 2)?;
                Ok(())
            })?;

            ctx.set_result(rows.len() as i64)?;
            Ok(())
        },
    )?;

    // vector_export_arrow(table_name, type) -> BLOB (Arrow IPC stream)
    //
    // Exports all vectors from the shadow data table as an Arrow IPC byte
    // buffer. The caller must supply the element type so blobs are decoded
    // correctly.
    db.create_scalar_function(
        "vector_export_arrow",
        &FunctionOptions::default().set_n_args(2),
        |ctx, args| {
            let table_name = args[0].get_str()?.to_owned();
            let type_name = args[1].get_str()?.to_owned();

            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;

            let db = ctx.db();

            // Collect all vector blobs from the data shadow table.
            let sql = ShadowOps::select_all_data_sql(&table_name);
            let mut stmt = db.prepare(&sql)?;
            stmt.query(())?;

            let mut blobs: Vec<Vec<u8>> = Vec::new();
            while let Some(row) = stmt.next()? {
                blobs.push(row[1].get_blob()?.to_vec());
            }

            if blobs.is_empty() {
                // Return an empty blob for an empty table.
                let empty: &[u8] = &[];
                ctx.set_result(empty)?;
                return Ok(());
            }

            let dim = blobs[0].len() / vtype.element_size();
            let ipc = arrow_io::vectors_to_arrow_ipc(&blobs, vtype, dim)
                .map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(&ipc[..])?;
            Ok(())
        },
    )?;

    // vector_insert_arrow(table_name, type, arrow_ipc_blob) -> INTEGER (row count)
    //
    // Imports vectors from an Arrow IPC blob into the shadow data table,
    // adding one row per vector. Returns the number of rows inserted.
    // Only inserts the vector column; metadata columns get NULL defaults.
    //
    // NOTE: Inserts directly into the shadow table, bypassing the in-memory
    // HNSW index. Call vector_rebuild_index afterwards to sync the index.
    db.create_scalar_function(
        "vector_insert_arrow",
        &FunctionOptions::default().set_n_args(3),
        |ctx, args| {
            let table_name = args[0].get_str()?.to_owned();
            let type_name = args[1].get_str()?.to_owned();
            let ipc_blob = args[2].get_blob()?.to_vec();

            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;

            if ipc_blob.is_empty() {
                ctx.set_result(0i64)?;
                return Ok(());
            }

            // Decode the Arrow IPC stream. We need the dimension, which we
            // infer from the first decoded vector.
            let blobs = arrow_io::arrow_ipc_to_vectors(&ipc_blob, vtype, 0)
                .map_err(|e| Error::Module(e.to_string()))?;

            if blobs.is_empty() {
                ctx.set_result(0i64)?;
                return Ok(());
            }

            let db = ctx.db();
            let insert_sql = ShadowOps::insert_vector_only_sql(&table_name);
            for blob in &blobs {
                db.insert(&insert_sql, [blob.as_slice()])?;
            }

            ctx.set_result(blobs.len() as i64)?;
            Ok(())
        },
    )?;

    // vector_sync_index(table_name) -> INTEGER (1 always: forces a persist of
    // the in-memory HNSW graph + row-count/max-rowid state, regardless of the
    // table's sync_every threshold).
    {
        let registry = registry.clone();
        db.create_scalar_function(
            "vector_sync_index",
            &FunctionOptions::default().set_n_args(1),
            move |ctx, args| {
                let name_arg = args[0].get_str()?.to_owned();
                let (state, config) = registry.get(&name_arg).map_err(Error::Module)?;

                // `persist_index` builds its shadow-table SQL from a bare
                // table name (unqualified by database), so it always writes
                // to the shadow tables in `main`. Persisting a table that
                // lives in a different attached database would silently
                // write into the wrong (or nonexistent) shadow tables in
                // `main`, so refuse that case explicitly rather than
                // qualifying the SQL — this keeps the fix scoped to
                // registry key/lookup disambiguation.
                if config.db_name != "main" {
                    return Err(Error::Module(format!(
                        "vector_sync_index is not supported for tables in attached database '{}' yet; only 'main' is supported",
                        config.db_name
                    )));
                }

                let mut s = state.borrow_mut();
                persist_index(ctx.db(), &config.table_name, &mut s)?;
                ctx.set_result(1i64)?;
                Ok(())
            },
        )?;
    }

    Ok(())
}
