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

/// Helper: decode blob to f64 values for any vector type
fn blob_to_f64s(blob: &[u8], vtype: VectorType) -> Vec<f64> {
    use crate::types::cast_blob;
    match vtype {
        VectorType::Float2 => cast_blob::<half::f16>(blob)
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        VectorType::Float4 => cast_blob::<f32>(blob).iter().map(|v| *v as f64).collect(),
        VectorType::Float8 => cast_blob::<f64>(blob).to_vec(),
        VectorType::Int1 => cast_blob::<i8>(blob).iter().map(|v| *v as f64).collect(),
        VectorType::Int2 => cast_blob::<i16>(blob).iter().map(|v| *v as f64).collect(),
        VectorType::Int4 => cast_blob::<i32>(blob).iter().map(|v| *v as f64).collect(),
    }
}

/// Helper: encode f64 values back to blob for any vector type
fn f64s_to_blob(values: &[f64], vtype: VectorType) -> Vec<u8> {
    use crate::types::slice_to_blob;
    match vtype {
        VectorType::Float2 => slice_to_blob(
            &values
                .iter()
                .map(|v| half::f16::from_f64(*v))
                .collect::<Vec<_>>(),
        ),
        VectorType::Float4 => slice_to_blob(&values.iter().map(|v| *v as f32).collect::<Vec<_>>()),
        VectorType::Float8 => slice_to_blob(values),
        VectorType::Int1 => slice_to_blob(&values.iter().map(|v| *v as i8).collect::<Vec<_>>()),
        VectorType::Int2 => slice_to_blob(&values.iter().map(|v| *v as i16).collect::<Vec<_>>()),
        VectorType::Int4 => slice_to_blob(&values.iter().map(|v| *v as i32).collect::<Vec<_>>()),
    }
}

/// Register a binary elementwise vector scalar function `name(a, b, type) ->
/// BLOB` that decodes both operands to f64 lanes, checks their dimensions
/// match, combines each lane pair with `op`, and re-encodes the result in
/// the input element type. Shared by `vector_add` and `vector_sub`, which
/// differ only in `op`.
fn register_binary_elementwise(db: &Connection, name: &str, op: fn(f64, f64) -> f64) -> Result<()> {
    db.create_scalar_function(
        name,
        &FunctionOptions::default()
            .set_n_args(3)
            .set_deterministic(true),
        move |ctx, args| {
            let type_name = args[2].get_str()?.to_owned();
            let a = args[0].get_blob()?.to_vec();
            let b = args[1].get_blob()?.to_vec();
            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            if a.len() != b.len() {
                return Err(Error::Module("vector dimensions do not match".into()));
            }
            let va = blob_to_f64s(&a, vtype);
            let vb = blob_to_f64s(&b, vtype);
            let out: Vec<f64> = va.iter().zip(&vb).map(|(x, y)| op(*x, *y)).collect();
            ctx.set_result(&f64s_to_blob(&out, vtype)[..])?;
            Ok(())
        },
    )
}

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
            let raw_arg = args[0].get_str()?.to_owned();
            // vector_rebuild_index is documented as bare-name only, but the
            // other table functions (vector_sync_index, vector_ef_search,
            // vector_index_info) all accept a `main.`-qualified name, so
            // accept the same prefix here for consistency (strip it — the
            // underlying shadow-table SQL is always unqualified/`main`).
            // Any other `db.` prefix is rejected with the same wording used
            // by those functions for attached databases.
            let table_name = if let Some(rest) = raw_arg.strip_prefix("main.") {
                rest.to_string()
            } else if let Some(dot) = raw_arg.find('.') {
                let db_name = &raw_arg[..dot];
                return Err(Error::Module(format!(
                    "vector_rebuild_index is not supported for tables in attached database '{db_name}' yet; only 'main' is supported"
                )));
            } else {
                raw_arg
            };
            let db = ctx.db();

            let meta = crate::vtab::load_meta_from_shadow(db, &table_name)?
                .ok_or_else(|| Error::Module(format!("no vector table named {table_name}")))?;
            if meta["mode"].as_str() == Some("exact") {
                return Err(Error::Module(format!(
                    "table '{table_name}' uses mode=exact and has no index"
                )));
            }
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

                if config.mode == crate::vtab::config::IndexMode::Exact {
                    return Err(Error::Module(format!(
                        "table '{}' uses mode=exact and has no index",
                        config.table_name
                    )));
                }

                let mut s = state.borrow_mut();
                persist_index(ctx.db(), &config.table_name, &mut s)?;
                ctx.set_result(1i64)?;
                Ok(())
            },
        )?;
    }

    // vector_ef_search(table_name, n) -> INTEGER (the new value)
    {
        let registry = registry.clone();
        db.create_scalar_function(
            "vector_ef_search",
            &FunctionOptions::default().set_n_args(2),
            move |ctx, args| {
                let table_name = args[0].get_str()?.to_owned();
                let n = args[1].get_i64();
                if n <= 0 {
                    return Err(Error::Module("ef_search must be positive".into()));
                }
                let (state, config) = registry
                    .get(&table_name)
                    .map_err(Error::Module)?;

                if config.db_name != "main" {
                    return Err(Error::Module(format!(
                        "vector_ef_search is not supported for tables in attached database '{}' yet; only 'main' is supported",
                        config.db_name
                    )));
                }

                {
                    let s = state.borrow();
                    if s.index.is_none() {
                        return Err(Error::Module(format!(
                            "table '{}' uses mode=exact and has no index",
                            table_name
                        )));
                    }
                }

                // Read-modify-write the persisted meta BEFORE mutating the
                // live index, so a failure here leaves the live index and
                // persisted meta consistent (both unchanged) rather than
                // diverging.
                let db = ctx.db();
                let mut meta = crate::vtab::load_meta_from_shadow(db, &config.table_name)?
                    .ok_or_else(|| {
                        Error::Module(format!("no vector table named {}", config.table_name))
                    })?;
                meta["ef_search"] = serde_json::json!(n);
                db.insert(&ShadowOps::upsert_index_sql(&config.table_name), |stmt: &mut query::Statement| {
                    "meta".bind_param(&mut *stmt, 1)?;
                    meta.to_string().as_str().bind_param(&mut *stmt, 2)?;
                    Ok(())
                })?;

                let s = state.borrow();
                let idx = s
                    .index
                    .as_ref()
                    .expect("index present: checked above under single-thread invariant");
                idx.set_ef_search(n as usize);
                drop(s);

                ctx.set_result(n)?;
                Ok(())
            },
        )?;
    }

    // vector_index_info(table_name) -> TEXT (JSON)
    {
        let registry = registry.clone();
        db.create_scalar_function(
            "vector_index_info",
            &FunctionOptions::default().set_n_args(1),
            move |ctx, args| {
                let table_name = args[0].get_str()?.to_owned();
                let (state, config) = registry
                    .get(&table_name)
                    .map_err(Error::Module)?;

                if config.db_name != "main" {
                    return Err(Error::Module(format!(
                        "vector_index_info is not supported for tables in attached database '{}' yet; only 'main' is supported",
                        config.db_name
                    )));
                }

                let s = state.borrow();
                let (rows, ef_live) = match &s.index {
                    Some(idx) => (idx.len() as i64, idx.ef_search() as i64),
                    None => {
                        let n: i64 = ctx.db().query_row(
                            &format!("SELECT count(*) FROM \"{}_data\"", config.table_name),
                            (),
                            |row| Ok(row[0].get_i64()),
                        )?;
                        (n, 0)
                    }
                };
                let info = serde_json::json!({
                    "rows": rows,
                    "dim": config.dim,
                    "type": config.vtype.name(),
                    "metric": config.metric.name(),
                    "mode": config.mode.name(),
                    "m": config.hnsw_params.m,
                    "ef_construction": config.hnsw_params.ef_construction,
                    "ef_search": if ef_live > 0 { ef_live } else { config.hnsw_params.ef_search as i64 },
                    "sync_every": config.sync_every,
                    "changes_since_persist": s.changes_since_persist,
                });
                ctx.set_result(info.to_string())?;
                Ok(())
            },
        )?;
    }

    // vector_normalize(blob, type) -> BLOB (float4 output for int inputs)
    db.create_scalar_function(
        "vector_normalize",
        &FunctionOptions::default()
            .set_n_args(2)
            .set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let blob = args[0].get_blob()?.to_vec();
            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let vals = blob_to_f64s(&blob, vtype);
            let norm = vals.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm == 0.0 {
                return Err(Error::Module("cannot normalize a zero vector".into()));
            }
            let out: Vec<f64> = vals.iter().map(|v| v / norm).collect();
            let out_type = if vtype.is_float() {
                vtype
            } else {
                VectorType::Float4
            };
            ctx.set_result(&f64s_to_blob(&out, out_type)[..])?;
            Ok(())
        },
    )?;

    // vector_add(a, b, type) -> BLOB
    register_binary_elementwise(db, "vector_add", |x, y| x + y)?;

    // vector_sub(a, b, type) -> BLOB
    register_binary_elementwise(db, "vector_sub", |x, y| x - y)?;

    // vector_scale(blob, factor, type) -> BLOB
    db.create_scalar_function(
        "vector_scale",
        &FunctionOptions::default()
            .set_n_args(3)
            .set_deterministic(true),
        |ctx, args| {
            let type_name = args[2].get_str()?.to_owned();
            let factor = args[1].get_f64();
            let blob = args[0].get_blob()?.to_vec();
            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let out: Vec<f64> = blob_to_f64s(&blob, vtype)
                .iter()
                .map(|v| v * factor)
                .collect();
            ctx.set_result(&f64s_to_blob(&out, vtype)[..])?;
            Ok(())
        },
    )?;

    // vector_slice(blob, type, start, end) -> BLOB (half-open element range)
    db.create_scalar_function(
        "vector_slice",
        &FunctionOptions::default()
            .set_n_args(4)
            .set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let start = args[2].get_i64();
            let end = args[3].get_i64();
            let blob = args[0].get_blob()?.to_vec();
            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let dim = (blob.len() / vtype.element_size()) as i64;
            if start < 0 || end < start || end > dim {
                return Err(Error::Module(format!(
                    "slice [{start}, {end}) out of bounds for dimension {dim}"
                )));
            }
            let es = vtype.element_size();
            let out = blob[start as usize * es..end as usize * es].to_vec();
            ctx.set_result(&out[..])?;
            Ok(())
        },
    )?;

    // vector_quantize_int8(blob, type) -> BLOB (int1), symmetric max-abs scaling
    db.create_scalar_function(
        "vector_quantize_int8",
        &FunctionOptions::default()
            .set_n_args(2)
            .set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let blob = args[0].get_blob()?.to_vec();
            let vtype =
                VectorType::from_name(&type_name).map_err(|e| Error::Module(e.to_string()))?;
            let vals = blob_to_f64s(&blob, vtype);
            let max_abs = vals.iter().fold(0f64, |m, v| m.max(v.abs()));
            let out: Vec<i8> = if max_abs == 0.0 {
                vec![0; vals.len()]
            } else {
                let scale = 127.0 / max_abs;
                vals.iter()
                    .map(|v| (v * scale).round().clamp(-127.0, 127.0) as i8)
                    .collect()
            };
            ctx.set_result(&crate::types::slice_to_blob(&out)[..])?;
            Ok(())
        },
    )?;

    Ok(())
}
