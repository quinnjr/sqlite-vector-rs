use sqlite3_ext::function::FunctionOptions;
use sqlite3_ext::*;

use crate::distance::{DistanceMetric, compute_distance};
use crate::json::{blob_to_json, json_to_blob};
use crate::types::VectorType;

/// Register all standalone scalar functions on a connection.
pub fn register_scalar_functions(db: &Connection) -> Result<()> {
    // vector_distance(blob_a, blob_b, metric, type) -> REAL
    db.create_scalar_function(
        "vector_distance",
        &FunctionOptions::default().set_n_args(4).set_deterministic(true),
        |ctx, args| {
            // Collect string args as owned values first to avoid borrow conflicts
            // with the blob borrows that follow.
            let metric_name = args[2].get_str()?.to_owned();
            let type_name = args[3].get_str()?.to_owned();
            let blob_a = args[0].get_blob()?.to_vec();
            let blob_b = args[1].get_blob()?.to_vec();

            let vtype = VectorType::from_name(&type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
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
        &FunctionOptions::default().set_n_args(2).set_deterministic(true),
        |ctx, args| {
            let json_text = args[0].get_str()?.to_owned();
            let type_name = args[1].get_str()?.to_owned();

            let vtype = VectorType::from_name(&type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let blob = json_to_blob(&json_text, vtype)
                .map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(&blob[..])?;
            Ok(())
        },
    )?;

    // vector_to_json(blob, type) -> TEXT
    db.create_scalar_function(
        "vector_to_json",
        &FunctionOptions::default().set_n_args(2).set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let blob = args[0].get_blob()?.to_vec();

            let vtype = VectorType::from_name(&type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let json = blob_to_json(&blob, vtype)
                .map_err(|e| Error::Module(e.to_string()))?;

            // Pass owned String — ToContextResult is implemented for String
            ctx.set_result(json)?;
            Ok(())
        },
    )?;

    // vector_dims(blob, type) -> INTEGER
    db.create_scalar_function(
        "vector_dims",
        &FunctionOptions::default().set_n_args(2).set_deterministic(true),
        |ctx, args| {
            let type_name = args[1].get_str()?.to_owned();
            let blob = args[0].get_blob()?;

            let vtype = VectorType::from_name(&type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let dims = blob.len() / vtype.element_size();

            ctx.set_result(dims as i64)?;
            Ok(())
        },
    )?;

    Ok(())
}
