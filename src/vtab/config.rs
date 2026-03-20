use std::fmt;

use crate::distance::DistanceMetric;
use crate::index::HnswParams;
use crate::types::VectorType;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config error: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

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
}

impl VectorTableConfig {
    pub fn parse(args: &[&str]) -> Result<Self, ConfigError> {
        if args.len() < 3 {
            return Err(ConfigError("expected at least module, db, and table name".into()));
        }

        let db_name = args[1].to_string();
        let table_name = args[2].to_string();

        let mut dim: Option<usize> = None;
        let mut vtype = VectorType::Float4;
        let mut metric = DistanceMetric::L2;
        let mut hnsw_params = HnswParams::default();
        let mut metadata_columns = Vec::new();

        for &arg in &args[3..] {
            let (key, value) = arg.split_once('=')
                .ok_or_else(|| ConfigError(format!("invalid argument: {arg}")))?;
            let key = key.trim();
            let value = value.trim().trim_matches('"');

            match key {
                "dim" => {
                    let d: i64 = value.parse()
                        .map_err(|_| ConfigError(format!("invalid dim: {value}")))?;
                    if d <= 0 {
                        return Err(ConfigError(format!("dim must be positive, got {d}")));
                    }
                    dim = Some(d as usize);
                }
                "type" => {
                    vtype = VectorType::from_name(value)
                        .map_err(|e| ConfigError(e.to_string()))?;
                }
                "metric" => {
                    metric = DistanceMetric::from_name(value)
                        .map_err(|e| ConfigError(e.to_string()))?;
                }
                "m" => {
                    hnsw_params.m = value.parse()
                        .map_err(|_| ConfigError(format!("invalid m: {value}")))?;
                }
                "ef_construction" => {
                    hnsw_params.ef_construction = value.parse()
                        .map_err(|_| ConfigError(format!("invalid ef_construction: {value}")))?;
                }
                "ef_search" => {
                    hnsw_params.ef_search = value.parse()
                        .map_err(|_| ConfigError(format!("invalid ef_search: {value}")))?;
                }
                "metadata" => {
                    metadata_columns = parse_metadata_columns(value)?;
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
        })
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
        let name = tokens.next()
            .ok_or_else(|| ConfigError("empty metadata column definition".to_string()))?
            .to_string();
        let sql_type = tokens.next()
            .ok_or_else(|| ConfigError(format!("missing type for metadata column {name}")))?
            .to_string();
        columns.push((name, sql_type));
    }
    Ok(columns)
}
