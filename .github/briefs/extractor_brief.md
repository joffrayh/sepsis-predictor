# Docstring Brief — `extraction/extractor.py`

**Module role**: Phase 1 of the pipeline. Reads raw MIMIC-IV `.csv.gz` files directly from disk using an in-process DuckDB connection (no PostgreSQL server required), executes 13 parameterised SQL queries from `extraction_metadata.json`, and writes the results as pipe-delimited CSVs to `data/extracted/`. Phase 1 is skipped entirely when `--raw-data-dir` is not passed to `main.py` — the checkpoint is at the CLI level, not inside this class.

---

## `MIMICExtractor` (class)

**Summary**: Stateful extractor that manages a single DuckDB in-process connection for its lifetime. On construction, registers every `.csv.gz` file under `hosp/` and `icu/` as a DuckDB view so that SQL queries can reference them as if they were database tables. The connection is held open across all `extract_table` calls and must be explicitly closed via `close()`.

**Constructor parameters**:
- `raw_data_dir` — path to the top-level MIMIC-IV directory (e.g. `data/raw/mimic-iv-3.1`). Must contain `hosp/` and `icu/` subdirectories of `.csv.gz` files.
- `export_dir` — directory where extracted CSVs are written. Created if it does not exist. Resolved to an absolute path at construction time.

**Invariants**:
- The DuckDB connection is in-process (`:memory:` style) — no socket, no server, no credentials.
- Views are registered under two schemas: `mimiciv_hosp.*` (for `hosp/*.csv.gz`) and `mimiciv_icu.*` (for `icu/*.csv.gz`). SQL in `extraction_metadata.json` must use these schema-qualified names.
- View names are derived from filenames: `chartevents.csv.gz` → `mimiciv_icu.chartevents`. The `.csv.gz` suffix is stripped; no other transformation is applied.

---

## `_register_views(self)`

**Summary**: Iterates all `.csv.gz` files in both `hosp/` and `icu/` subdirectories and registers each as a DuckDB `CREATE VIEW ... AS SELECT * FROM read_csv_auto(...)`. Called once in `__init__`.

**Side effects**: Creates `mimiciv_hosp` and `mimiciv_icu` schemas and one view per file. Prints a confirmation line per registered view.

**Invariants**:
- Paths are converted to forward-slash absolute paths before being embedded in SQL (Windows compatibility).
- Any `.csv.gz` file present in the directories is registered, regardless of whether it is queried by `extraction_metadata.json`. Unqueried views are harmless.

---

## `_prepare_query(self, query)`

**Summary**: Sanitises a raw SQL string from `extraction_metadata.json` so it can be used inside a DuckDB `COPY (...) TO ...` statement. Strips the PostgreSQL-specific `CREATE TEMP TABLE <name> AS` prefix (used by the `demog` query, which was originally written for PostgreSQL) and any trailing semicolons.

**Parameters**:
- `query` — raw SQL string as stored in `extraction_metadata.json`. May be a plain SELECT, a CTE (`WITH ... SELECT`), or a `CREATE TEMP TABLE ... AS SELECT ...`.

**Returns**: Clean SELECT/CTE string with no leading DDL and no trailing semicolon.

**Invariants**:
- The regex is case-insensitive and anchored to the start of the string (`^\s*`).
- Only the `CREATE TEMP TABLE ... AS` prefix is stripped — no other SQL transformation is applied. This is a single-purpose compatibility shim for the one legacy query in the metadata.

---

## `_run_with_timer(self, query, output_path)`

**Summary**: Wraps a DuckDB `COPY` export in a background ticker thread that prints a live elapsed-time and write-speed display to the terminal. The actual export runs in the calling thread; the ticker daemon thread samples file size every 0.4 s and computes a rolling 5-sample MB/s estimate.

**Parameters**:
- `query` — prepared SELECT query (output of `_prepare_query`).
- `output_path` — absolute path for the output CSV. Must use forward slashes (DuckDB requirement on all platforms).

**Side effects**:
- Writes the pipe-delimited CSV to `output_path` via `COPY (...) TO ... (HEADER, DELIMITER '|')`.
- Prints live progress to stdout using `\r` overwrite. Prints a final "done in Xs | Y MB" summary line.
- The ticker thread is a daemon — it is guaranteed to be joined before the function returns even if an exception is raised (via `finally: done.set(); t.join()`).

**Invariants**:
- MB/s display is suppressed until at least 2 samples with non-zero written bytes exist (avoids division-by-zero and spurious 0 MB/s on fast queries that buffer before first flush).
- The DuckDB `COPY` statement always includes `HEADER` (column names on first row) and `DELIMITER '|'` (pipe separator). All downstream readers must use `sep="|"`.

---

## `extract_all(self, metadata_path, tables)`

**Summary**: Loads `extraction_metadata.json` and calls `extract_table` for each entry. Optionally restricts extraction to a named subset of tables.

**Parameters**:
- `metadata_path` — path to `extraction_metadata.json`. Default matches the repo layout.
- `tables` — optional list of key names (e.g. `["chartevents", "labs_ce"]`). If `None`, all 13 tables are extracted.

**Side effects**: Prints progress headers per table. Delegates all I/O to `extract_table`.

**Invariants**: Table order follows JSON insertion order (Python 3.7+ dict ordering). This is not meaningful — tables are independent and could be extracted in any order.

---

## `extract_table(self, name, conf, index, total)`

**Summary**: Extracts a single table to a pipe-delimited CSV. Skips silently if the output file already exists (per-table checkpoint — allows re-running Phase 1 after a partial failure without re-extracting completed tables).

**Parameters**:
- `name` — logical table name (for logging only).
- `conf` — dict entry from `extraction_metadata.json`. Required key: `"query"` (SQL string), `"file_name"` (output filename without `.csv` extension).
- `index`, `total` — optional 1-based position in the extraction sequence, used for `[N/M]` log prefix. Pass `None` for both when calling outside `extract_all`.

**Side effects**: Writes `{export_dir}/{conf["file_name"]}.csv` (pipe-delimited). Prints status to stdout.

**Invariants**:
- The skip-if-exists check uses `os.path.exists` on the final `.csv` path. A partial write from a previous interrupted run would not be detected — the partially written file would be treated as complete. If extraction results look wrong, manually delete the relevant CSV and re-run.
- Output filename is always `{file_name}.csv` — the `.csv` extension is added here, not in the metadata.

---

## `close(self)`

**Summary**: Closes the DuckDB connection. Must be called when extraction is complete to release file handles and in-process DuckDB resources.

**Invariants**: Called in a `finally` block in `main.py` to ensure the connection is closed even if extraction raises an exception.
