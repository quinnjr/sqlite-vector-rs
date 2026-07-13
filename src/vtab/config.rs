use std::fmt;

use crate::distance::DistanceMetric;
use crate::index::{HnswIndex, HnswParams, IndexError};
use crate::types::VectorType;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config error: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Whether a table is served by an in-memory HNSW index or by a streaming
/// brute-force exact scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    Hnsw,
    Exact,
}

impl IndexMode {
    pub fn from_name(name: &str) -> Result<Self, ConfigError> {
        match name {
            "hnsw" => Ok(Self::Hnsw),
            "exact" => Ok(Self::Exact),
            other => Err(ConfigError(format!("unknown mode: {other}"))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Hnsw => "hnsw",
            Self::Exact => "exact",
        }
    }
}

/// Parsed configuration from CREATE VIRTUAL TABLE arguments.
#[derive(Debug, Clone)]
pub struct VectorTableConfig {
    pub db_name: String,
    pub table_name: String,
    pub dim: usize,
    pub vtype: VectorType,
    pub metric: DistanceMetric,
    pub hnsw_params: HnswParams,
    pub metadata_columns: Vec<(String, String)>,
    pub sync_every: u64,
    pub mode: IndexMode,
}

impl VectorTableConfig {
    pub fn parse(args: &[&str]) -> Result<Self, ConfigError> {
        if args.len() < 3 {
            return Err(ConfigError(
                "expected at least module, db, and table name".into(),
            ));
        }

        let db_name = args[1].to_string();
        let table_name = args[2].to_string();

        let mut dim: Option<usize> = None;
        let mut vtype = VectorType::Float4;
        let mut metric = DistanceMetric::L2;
        let mut hnsw_params = HnswParams::default();
        let mut metadata_columns = Vec::new();
        let mut sync_every: u64 = 1024;
        let mut mode = IndexMode::Hnsw;

        for &arg in &args[3..] {
            let (key, value) = arg
                .split_once('=')
                .ok_or_else(|| ConfigError(format!("invalid argument: {arg}")))?;
            let key = key.trim();
            let value = value.trim().trim_matches('"');

            match key {
                "dim" => {
                    let d: i64 = value
                        .parse()
                        .map_err(|_| ConfigError(format!("invalid dim: {value}")))?;
                    if d <= 0 {
                        return Err(ConfigError(format!("dim must be positive, got {d}")));
                    }
                    dim = Some(d as usize);
                }
                "type" => {
                    vtype = VectorType::from_name(value).map_err(|e| ConfigError(e.to_string()))?;
                }
                "metric" => {
                    metric =
                        DistanceMetric::from_name(value).map_err(|e| ConfigError(e.to_string()))?;
                }
                "m" => {
                    hnsw_params.m = value
                        .parse()
                        .map_err(|_| ConfigError(format!("invalid m: {value}")))?;
                }
                "ef_construction" => {
                    hnsw_params.ef_construction = value
                        .parse()
                        .map_err(|_| ConfigError(format!("invalid ef_construction: {value}")))?;
                }
                "ef_search" => {
                    hnsw_params.ef_search = value
                        .parse()
                        .map_err(|_| ConfigError(format!("invalid ef_search: {value}")))?;
                }
                "metadata" => {
                    metadata_columns = parse_metadata_columns(value)?;
                }
                "sync_every" => {
                    let n: u64 = value
                        .parse()
                        .map_err(|_| ConfigError(format!("invalid sync_every: {value}")))?;
                    if n == 0 {
                        return Err(ConfigError("sync_every must be >= 1".into()));
                    }
                    sync_every = n;
                }
                "mode" => {
                    mode = IndexMode::from_name(value)?;
                }
                other => {
                    return Err(ConfigError(format!("unknown parameter: {other}")));
                }
            }
        }

        let dim = dim.ok_or_else(|| ConfigError("dim is required".into()))?;

        Ok(Self {
            db_name,
            table_name,
            dim,
            vtype,
            metric,
            hnsw_params,
            metadata_columns,
            sync_every,
            mode,
        })
    }

    pub fn to_meta_json(&self) -> String {
        serde_json::json!({
            "dim": self.dim,
            "type": self.vtype.name(),
            "metric": self.metric.name(),
            "m": self.hnsw_params.m,
            "ef_construction": self.hnsw_params.ef_construction,
            "ef_search": self.hnsw_params.ef_search,
            "sync_every": self.sync_every,
            "mode": self.mode.name(),
        })
        .to_string()
    }

    pub fn params_from_meta(
        meta: &serde_json::Value,
    ) -> Result<(usize, VectorType, DistanceMetric, HnswParams), ConfigError> {
        let dim = meta["dim"]
            .as_u64()
            .ok_or_else(|| ConfigError("meta missing dim".into()))? as usize;
        let vtype = VectorType::from_name(
            meta["type"]
                .as_str()
                .ok_or_else(|| ConfigError("meta missing type".into()))?,
        )
        .map_err(|e| ConfigError(e.to_string()))?;
        let metric = DistanceMetric::from_name(
            meta["metric"]
                .as_str()
                .ok_or_else(|| ConfigError("meta missing metric".into()))?,
        )
        .map_err(|e| ConfigError(e.to_string()))?;
        let m = meta["m"]
            .as_u64()
            .ok_or_else(|| ConfigError("meta missing m".into()))? as usize;
        let ef_construction = meta["ef_construction"]
            .as_u64()
            .ok_or_else(|| ConfigError("meta missing ef_construction".into()))?
            as usize;
        let ef_search = meta["ef_search"]
            .as_u64()
            .ok_or_else(|| ConfigError("meta missing ef_search".into()))?
            as usize;
        let params = HnswParams {
            m,
            ef_construction,
            ef_search,
        };
        Ok((dim, vtype, metric, params))
    }

    /// Construct a fresh, empty HNSW index matching this config's
    /// dim/type/metric/HNSW params. Shared by every "rebuild the index from
    /// `_data`" path (reconcile, rollback, savepoint rollback-to) so they
    /// can't drift out of sync with each other.
    pub(crate) fn new_index(&self) -> Result<HnswIndex, IndexError> {
        HnswIndex::new(self.dim, self.vtype, self.metric, Some(self.hnsw_params))
    }

    pub fn vtab_schema(&self) -> String {
        let mut cols = vec![
            "id INTEGER PRIMARY KEY".to_string(),
            "vector BLOB".to_string(),
        ];
        for (name, sql_type) in &self.metadata_columns {
            cols.push(format!("{name} {sql_type}"));
        }
        cols.push("distance REAL HIDDEN".to_string());
        format!("CREATE TABLE x({})", cols.join(", "))
    }
}

fn parse_metadata_columns(spec: &str) -> Result<Vec<(String, String)>, ConfigError> {
    let mut columns = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut tokens = part.split_whitespace();
        let name = tokens
            .next()
            .ok_or_else(|| ConfigError("empty metadata column definition".to_string()))?
            .to_string();
        let sql_type = tokens
            .next()
            .ok_or_else(|| ConfigError(format!("missing type for metadata column {name}")))?
            .to_string();
        columns.push((name, sql_type));
    }
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::types::VectorType;

    // ----------------------------------------------------------------
    // parse — minimal args (dim only, defaults for type/metric)
    // ----------------------------------------------------------------

    #[test]
    fn parse_minimal_args_uses_defaults() {
        let cfg = VectorTableConfig::parse(&["vector", "main", "test", "dim=3"]).unwrap();
        assert_eq!(cfg.db_name, "main");
        assert_eq!(cfg.table_name, "test");
        assert_eq!(cfg.dim, 3);
        assert_eq!(cfg.vtype, VectorType::Float4);
        assert_eq!(cfg.metric, DistanceMetric::L2);
        // HNSW defaults
        assert_eq!(cfg.hnsw_params.m, 16);
        assert_eq!(cfg.hnsw_params.ef_construction, 200);
        assert_eq!(cfg.hnsw_params.ef_search, 64);
        assert!(cfg.metadata_columns.is_empty());
    }

    // ----------------------------------------------------------------
    // parse — all parameters specified
    // ----------------------------------------------------------------

    #[test]
    fn parse_all_params_specified() {
        let cfg = VectorTableConfig::parse(&[
            "vector",
            "main",
            "embeddings",
            "dim=128",
            "type=float4",
            "metric=l2",
            "m=32",
            "ef_construction=400",
            "ef_search=128",
        ])
        .unwrap();
        assert_eq!(cfg.dim, 128);
        assert_eq!(cfg.vtype, VectorType::Float4);
        assert_eq!(cfg.metric, DistanceMetric::L2);
        assert_eq!(cfg.hnsw_params.m, 32);
        assert_eq!(cfg.hnsw_params.ef_construction, 400);
        assert_eq!(cfg.hnsw_params.ef_search, 128);
    }

    // ----------------------------------------------------------------
    // parse — type=float8, metric=cosine
    // ----------------------------------------------------------------

    #[test]
    fn parse_float8_cosine() {
        let cfg = VectorTableConfig::parse(&[
            "vector",
            "main",
            "vecs",
            "dim=64",
            "type=float8",
            "metric=cosine",
        ])
        .unwrap();
        assert_eq!(cfg.vtype, VectorType::Float8);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
    }

    // ----------------------------------------------------------------
    // parse — custom HNSW params (m, ef_construction, ef_search)
    // ----------------------------------------------------------------

    #[test]
    fn parse_custom_hnsw_params() {
        let cfg = VectorTableConfig::parse(&[
            "vector",
            "main",
            "idx",
            "dim=16",
            "m=8",
            "ef_construction=100",
            "ef_search=50",
        ])
        .unwrap();
        assert_eq!(cfg.hnsw_params.m, 8);
        assert_eq!(cfg.hnsw_params.ef_construction, 100);
        assert_eq!(cfg.hnsw_params.ef_search, 50);
    }

    // ----------------------------------------------------------------
    // parse — metadata columns
    // ----------------------------------------------------------------

    #[test]
    fn parse_metadata_columns_parsed_correctly() {
        let cfg = VectorTableConfig::parse(&[
            "vector",
            "main",
            "docs",
            "dim=4",
            "metadata=label TEXT,score REAL",
        ])
        .unwrap();
        assert_eq!(cfg.metadata_columns.len(), 2);
        assert_eq!(
            cfg.metadata_columns[0],
            ("label".to_string(), "TEXT".to_string())
        );
        assert_eq!(
            cfg.metadata_columns[1],
            ("score".to_string(), "REAL".to_string())
        );
    }

    // ----------------------------------------------------------------
    // parse errors — too few args (< 3)
    // ----------------------------------------------------------------

    #[test]
    fn parse_error_too_few_args_zero() {
        let err = VectorTableConfig::parse(&[]).unwrap_err();
        assert!(
            err.0.contains("at least"),
            "unexpected error message: {}",
            err.0
        );
    }

    #[test]
    fn parse_error_too_few_args_two() {
        // Only module + db name; table name is absent.
        let err = VectorTableConfig::parse(&["vector", "main"]).unwrap_err();
        assert!(
            err.0.contains("at least"),
            "unexpected error message: {}",
            err.0
        );
    }

    // ----------------------------------------------------------------
    // parse errors — missing dim
    // ----------------------------------------------------------------

    #[test]
    fn parse_error_missing_dim() {
        let err = VectorTableConfig::parse(&["vector", "main", "tbl", "type=float4"]).unwrap_err();
        assert!(
            err.0.contains("dim"),
            "expected error mentioning 'dim', got: {}",
            err.0
        );
    }

    // ----------------------------------------------------------------
    // parse errors — invalid dim values
    // ----------------------------------------------------------------

    #[test]
    fn parse_error_dim_zero() {
        let err = VectorTableConfig::parse(&["vector", "main", "tbl", "dim=0"]).unwrap_err();
        assert!(
            err.0.contains("positive") || err.0.contains("dim"),
            "unexpected error: {}",
            err.0
        );
    }

    #[test]
    fn parse_error_dim_negative() {
        let err = VectorTableConfig::parse(&["vector", "main", "tbl", "dim=-5"]).unwrap_err();
        assert!(
            err.0.contains("positive") || err.0.contains("dim"),
            "unexpected error: {}",
            err.0
        );
    }

    #[test]
    fn parse_error_dim_non_numeric() {
        let err = VectorTableConfig::parse(&["vector", "main", "tbl", "dim=abc"]).unwrap_err();
        assert!(
            err.0.contains("dim"),
            "expected error mentioning 'dim', got: {}",
            err.0
        );
    }

    // ----------------------------------------------------------------
    // parse errors — unknown parameter
    // ----------------------------------------------------------------

    #[test]
    fn parse_error_unknown_parameter() {
        let err =
            VectorTableConfig::parse(&["vector", "main", "tbl", "dim=4", "foo=bar"]).unwrap_err();
        assert!(
            err.0.contains("unknown") && err.0.contains("foo"),
            "unexpected error: {}",
            err.0
        );
    }

    // ----------------------------------------------------------------
    // parse errors — arg without '='
    // ----------------------------------------------------------------

    #[test]
    fn parse_error_arg_without_equals() {
        let err = VectorTableConfig::parse(&["vector", "main", "tbl", "dim=4", "invalidarg"])
            .unwrap_err();
        assert!(
            err.0.contains("invalid argument") || err.0.contains("invalidarg"),
            "unexpected error: {}",
            err.0
        );
    }

    // ----------------------------------------------------------------
    // vtab_schema — no metadata
    // ----------------------------------------------------------------

    #[test]
    fn vtab_schema_no_metadata() {
        let cfg = VectorTableConfig::parse(&["vector", "main", "tbl", "dim=3"]).unwrap();
        let schema = cfg.vtab_schema();
        assert_eq!(
            schema,
            "CREATE TABLE x(id INTEGER PRIMARY KEY, vector BLOB, distance REAL HIDDEN)"
        );
    }

    // ----------------------------------------------------------------
    // vtab_schema — with metadata columns
    // ----------------------------------------------------------------

    #[test]
    fn vtab_schema_with_metadata_columns_before_distance() {
        let cfg = VectorTableConfig::parse(&[
            "vector",
            "main",
            "tbl",
            "dim=4",
            "metadata=label TEXT,score REAL",
        ])
        .unwrap();
        let schema = cfg.vtab_schema();
        assert_eq!(
            schema,
            "CREATE TABLE x(id INTEGER PRIMARY KEY, vector BLOB, label TEXT, score REAL, distance REAL HIDDEN)"
        );
        // Verify ordering: metadata must appear before distance.
        let label_pos = schema.find("label").unwrap();
        let distance_pos = schema.find("distance").unwrap();
        assert!(
            label_pos < distance_pos,
            "metadata columns must precede distance in schema"
        );
    }
}
