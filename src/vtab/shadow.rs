use crate::vtab::config::VectorTableConfig;

/// SQL statements for shadow table management.
pub struct ShadowOps;

impl ShadowOps {
    pub fn create_data_table_sql(config: &VectorTableConfig) -> String {
        let mut cols = vec![
            "id INTEGER PRIMARY KEY AUTOINCREMENT".to_string(),
            "vector BLOB NOT NULL".to_string(),
        ];
        for (name, sql_type) in &config.metadata_columns {
            cols.push(format!("{name} {sql_type}"));
        }
        format!(
            "CREATE TABLE IF NOT EXISTS \"{}_data\"({})",
            config.table_name,
            cols.join(", ")
        )
    }

    pub fn create_index_table_sql(config: &VectorTableConfig) -> String {
        format!(
            "CREATE TABLE IF NOT EXISTS \"{}_index\"(key TEXT PRIMARY KEY, value BLOB)",
            config.table_name
        )
    }

    pub fn drop_shadow_tables_sql(table_name: &str) -> Vec<String> {
        vec![
            format!("DROP TABLE IF EXISTS \"{table_name}_data\""),
            format!("DROP TABLE IF EXISTS \"{table_name}_index\""),
        ]
    }

    pub fn insert_data_sql(config: &VectorTableConfig) -> String {
        let mut col_names = vec!["vector".to_string()];
        let mut placeholders = vec!["?".to_string()];
        for (name, _) in &config.metadata_columns {
            col_names.push(name.clone());
            placeholders.push("?".to_string());
        }
        format!(
            "INSERT INTO \"{}_data\"({}) VALUES({})",
            config.table_name,
            col_names.join(", "),
            placeholders.join(", ")
        )
    }

    pub fn insert_vector_only_sql(table_name: &str) -> String {
        format!("INSERT INTO \"{table_name}_data\"(vector) VALUES(?)")
    }

    pub fn delete_data_sql(table_name: &str) -> String {
        format!("DELETE FROM \"{table_name}_data\" WHERE id = ?")
    }

    pub fn select_data_sql(table_name: &str) -> String {
        format!("SELECT * FROM \"{table_name}_data\" WHERE id = ?")
    }

    pub fn select_all_data_sql(table_name: &str) -> String {
        format!("SELECT * FROM \"{table_name}_data\"")
    }

    pub fn upsert_index_sql(table_name: &str) -> String {
        format!("INSERT OR REPLACE INTO \"{table_name}_index\"(key, value) VALUES(?, ?)")
    }

    pub fn select_index_sql(table_name: &str) -> String {
        format!("SELECT value FROM \"{table_name}_index\" WHERE key = ?")
    }
}
