use std::io::{self, BufRead, IsTerminal, Write};

use rusqlite::Connection;
use rusqlite::types::Value;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args
        .iter()
        .any(|a| a == "-h" || a == "--help" || a == "-help")
    {
        print_usage(&args[0]);
        return;
    }

    if args.iter().any(|a| a == "--version" || a == "-version") {
        println!(
            "sqlite3-vector {} (SQLite {})",
            env!("CARGO_PKG_VERSION"),
            rusqlite::version()
        );
        return;
    }

    let db_path = args.get(1).filter(|a| !a.starts_with('-'));

    let conn = match db_path {
        Some(path) => Connection::open(path).unwrap_or_else(|e| {
            eprintln!("Error: cannot open \"{path}\": {e}");
            std::process::exit(1);
        }),
        None => Connection::open_in_memory().unwrap_or_else(|e| {
            eprintln!("Error: cannot create in-memory database: {e}");
            std::process::exit(1);
        }),
    };

    if let Err(e) = sqlite_vector_rs::register(&conn) {
        eprintln!("Warning: failed to load vector extension: {e}");
        eprintln!("Vector operations will not be available.");
    }

    let interactive = io::stdin().is_terminal();

    if interactive {
        println!(
            "sqlite3-vector v{} (SQLite {})",
            env!("CARGO_PKG_VERSION"),
            rusqlite::version()
        );
        println!("Enter \".help\" for usage hints.");
        if db_path.is_none() {
            println!("Connected to a transient in-memory database.");
            println!("Use \".open FILENAME\" to reopen on a persistent database.");
        }
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut state = ReplState::default();

    if interactive {
        print_prompt(&mut stdout, true);
    }

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        state.buffer.push_str(&line);

        if state.buffer.trim().is_empty() {
            state.buffer.clear();
            if interactive {
                print_prompt(&mut stdout, true);
            }
            continue;
        }

        if state.buffer.trim_start().starts_with('.') {
            let cmd = state.buffer.trim().to_string();
            state.buffer.clear();

            if cmd == ".quit" || cmd == ".exit" {
                break;
            }

            handle_dot_command(&conn, &cmd, &mut state);
            if interactive {
                print_prompt(&mut stdout, true);
            }
            continue;
        }

        if !state.buffer.trim_end().ends_with(';') {
            state.buffer.push('\n');
            if interactive {
                print_prompt(&mut stdout, false);
            }
            continue;
        }

        let sql = state.buffer.trim().to_string();
        state.buffer.clear();

        execute_sql(&conn, &sql, &state);
        if interactive {
            print_prompt(&mut stdout, true);
        }
    }

    if interactive {
        println!();
    }
}

fn print_usage(program: &str) {
    println!("Usage: {program} [OPTIONS] [DATABASE]");
    println!();
    println!("SQLite shell with the sqlite-vector-rs extension pre-loaded.");
    println!();
    println!("Arguments:");
    println!("  DATABASE        Database file to open (in-memory if omitted)");
    println!();
    println!("Options:");
    println!("  -h, --help      Show this help message");
    println!("  --version       Show version information");
}

#[derive(Clone, Copy)]
enum OutputMode {
    Column,
    Csv,
    Line,
    List,
}

struct ReplState {
    buffer: String,
    headers: bool,
    mode: OutputMode,
}

impl Default for ReplState {
    fn default() -> Self {
        Self {
            buffer: String::new(),
            headers: true,
            mode: OutputMode::Column,
        }
    }
}

fn print_prompt(stdout: &mut io::Stdout, primary: bool) {
    if primary {
        print!("sqlite3-vector> ");
    } else {
        print!("          ...> ");
    }
    let _ = stdout.flush();
}

fn execute_sql(conn: &Connection, sql: &str, state: &ReplState) {
    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };

    let col_count = stmt.column_count();

    if col_count == 0 {
        match stmt.execute(()) {
            Ok(_) => {}
            Err(e) => eprintln!("Error: {e}"),
        }
        return;
    }

    let col_names: Vec<String> = (0..col_count)
        .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
        .collect();

    let rows = match stmt.query_map((), |row| {
        let mut values = Vec::with_capacity(col_count);
        for i in 0..col_count {
            let val: Value = row.get(i).unwrap_or(Value::Null);
            values.push(val);
        }
        Ok(values)
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };

    let mut all_rows: Vec<Vec<String>> = Vec::new();
    for row in rows {
        match row {
            Ok(values) => {
                let strings: Vec<String> = values.iter().map(format_value).collect();
                all_rows.push(strings);
            }
            Err(e) => {
                eprintln!("Error: {e}");
                return;
            }
        }
    }

    match state.mode {
        OutputMode::Column => print_column(&col_names, &all_rows, state.headers),
        OutputMode::Csv => print_csv(&col_names, &all_rows, state.headers),
        OutputMode::Line => print_line(&col_names, &all_rows),
        OutputMode::List => print_list(&col_names, &all_rows, state.headers),
    }
}

fn format_value(val: &Value) -> String {
    match val {
        Value::Null => "NULL".to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Real(f) => format!("{f}"),
        Value::Text(s) => s.clone(),
        Value::Blob(b) => {
            if b.len() <= 20 {
                format!("X'{}'", hex_encode(b))
            } else {
                format!("X'{}...' ({} bytes)", hex_encode(&b[..16]), b.len())
            }
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

fn print_column(col_names: &[String], rows: &[Vec<String>], show_headers: bool) {
    let num_cols = col_names.len();
    let mut widths: Vec<usize> = col_names.iter().map(|h| h.len()).collect();

    for row in rows {
        for (i, val) in row.iter().enumerate() {
            if i < num_cols {
                widths[i] = widths[i].max(val.len());
            }
        }
    }

    if show_headers {
        let header: String = col_names
            .iter()
            .enumerate()
            .map(|(i, h)| format!("{:<w$}", h, w = widths[i]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{header}");

        let sep: String = widths
            .iter()
            .map(|w| "-".repeat(*w))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{sep}");
    }

    for row in rows {
        let line: String = row
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let w = widths.get(i).copied().unwrap_or(v.len());
                format!("{:<w$}", v)
            })
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
    }
}

fn print_csv(col_names: &[String], rows: &[Vec<String>], show_headers: bool) {
    if show_headers {
        println!("{}", col_names.join(","));
    }
    for row in rows {
        let escaped: Vec<String> = row
            .iter()
            .map(|v| {
                if v.contains(',') || v.contains('"') || v.contains('\n') {
                    format!("\"{}\"", v.replace('"', "\"\""))
                } else {
                    v.clone()
                }
            })
            .collect();
        println!("{}", escaped.join(","));
    }
}

fn print_line(col_names: &[String], rows: &[Vec<String>]) {
    let max_w = col_names.iter().map(|h| h.len()).max().unwrap_or(0);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            println!();
        }
        for (j, val) in row.iter().enumerate() {
            let header = col_names.get(j).map(|s| s.as_str()).unwrap_or("?");
            println!("{:>w$} = {val}", header, w = max_w);
        }
    }
}

fn print_list(col_names: &[String], rows: &[Vec<String>], show_headers: bool) {
    if show_headers {
        println!("{}", col_names.join("|"));
    }
    for row in rows {
        println!("{}", row.join("|"));
    }
}

fn handle_dot_command(conn: &Connection, cmd: &str, state: &mut ReplState) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    let command = parts.first().copied().unwrap_or("");

    match command {
        ".help" => {
            println!(".exit                  Exit this program");
            println!(".headers on|off        Turn display of headers on or off");
            println!(".help                  Show this help");
            println!(".mode MODE             Set output mode (column, csv, line, list)");
            println!(".quit                  Exit this program");
            println!(".schema ?TABLE?        Show CREATE statements");
            println!(".tables ?PATTERN?      List names of tables matching LIKE pattern");
        }
        ".tables" => {
            let sql = match parts.get(1) {
                Some(p) => format!(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{}' ORDER BY 1;",
                    p.replace('\'', "''")
                ),
                None => "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1;".to_string(),
            };
            let list_state = ReplState {
                headers: false,
                mode: OutputMode::List,
                ..Default::default()
            };
            execute_sql(conn, &sql, &list_state);
        }
        ".schema" => {
            let sql = match parts.get(1) {
                Some(t) => format!(
                    "SELECT sql FROM sqlite_master WHERE name='{}' ORDER BY 1;",
                    t.replace('\'', "''")
                ),
                None => "SELECT sql FROM sqlite_master ORDER BY 1;".to_string(),
            };
            let list_state = ReplState {
                headers: false,
                mode: OutputMode::List,
                ..Default::default()
            };
            execute_sql(conn, &sql, &list_state);
        }
        ".headers" => match parts.get(1) {
            Some(&"on") => state.headers = true,
            Some(&"off") => state.headers = false,
            _ => eprintln!("Usage: .headers on|off"),
        },
        ".mode" => match parts.get(1) {
            Some(&"column") => state.mode = OutputMode::Column,
            Some(&"csv") => state.mode = OutputMode::Csv,
            Some(&"line") => state.mode = OutputMode::Line,
            Some(&"list") => state.mode = OutputMode::List,
            _ => eprintln!("Usage: .mode column|csv|line|list"),
        },
        _ => {
            eprintln!("Error: unknown command: \"{cmd}\"");
            eprintln!("Use \".help\" for available commands.");
        }
    }
}
