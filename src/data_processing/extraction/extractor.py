import glob
import json
import os
import re
import threading
import time

import duckdb


class MIMICExtractor:
    """
    Unified extractor for MIMIC-IV raw CSV files using DuckDB.

    Reads directly from the raw ``.csv.gz`` files — no running database
    server required.  Each MIMIC-IV table is registered as a DuckDB view
    under the ``mimiciv_hosp`` or ``mimiciv_icu`` schema so that the SQL
    queries in ``extraction_metadata.json`` work without modification.
    """

    def __init__(
        self,
        raw_data_dir="data/raw/mimic-iv-3.1",
        export_dir="data/processed_files",
    ):
        """
        Initialise the extractor and register all MIMIC-IV tables as views.

        Parameters
        ----------
        raw_data_dir : str
            Path to the top-level MIMIC-IV directory containing ``hosp/`` and
            ``icu/`` subdirectories of ``.csv.gz`` files.
        export_dir : str
            Directory where extracted pipe-delimited ``.csv`` files are written.
        """
        self.raw_data_dir = raw_data_dir
        self.export_dir = os.path.join(os.getcwd(), export_dir)
        os.makedirs(self.export_dir, exist_ok=True)

        self.conn = duckdb.connect()
        self._register_views()

    def _register_views(self):
        """Register each MIMIC-IV table as a DuckDB view under its schema."""
        schema_subdirs = {
            "mimiciv_hosp": os.path.join(self.raw_data_dir, "hosp"),
            "mimiciv_icu": os.path.join(self.raw_data_dir, "icu"),
        }
        for schema, directory in schema_subdirs.items():
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            for gz_path in glob.glob(os.path.join(directory, "*.csv.gz")):
                table_name = os.path.basename(gz_path).replace(".csv.gz", "")
                abs_path = os.path.abspath(gz_path).replace("\\", "/")
                self.conn.execute(
                    f"CREATE VIEW {schema}.{table_name} AS "
                    f"SELECT * FROM read_csv_auto('{abs_path}')"
                )
                print(f"  Registered view: {schema}.{table_name}")

    def _prepare_query(self, query):
        """
        Normalise a query for DuckDB execution.

        Strips the PostgreSQL-specific ``CREATE TEMP TABLE ... AS`` prefix
        used by the ``demog`` query and any trailing semicolons, leaving a
        plain SELECT / CTE that can be wrapped in a DuckDB COPY statement.

        Parameters
        ----------
        query : str
            Raw SQL string from ``extraction_metadata.json``.

        Returns
        -------
        str
            Clean SELECT query suitable for ``COPY (...) TO ...``.
        """
        query = re.sub(
            r"^\s*CREATE\s+TEMP\s+TABLE\s+\w+\s+AS\s*",
            "",
            query,
            flags=re.IGNORECASE,
        )
        return query.rstrip(";").strip()

    def _run_with_timer(self, query, output_path):
        """
        Execute a DuckDB COPY query with a live elapsed-time and throughput display.

        The ticker samples the output file size every 0.4 s and computes a
        rolling write speed over a 5-sample window.  Both ``MB written`` and
        ``MB/s`` are suppressed until the first bytes appear on disk (DuckDB
        may buffer briefly before the first flush).

        Parameters
        ----------
        query : str
            Prepared SELECT query to execute.
        output_path : str
            Absolute path for the output CSV file.
        """
        done = threading.Event()
        start = time.time()
        spinner = "|/-\\"
        _window = 5

        def _ticker():
            i = 0
            samples = []
            while not done.is_set():
                elapsed = time.time() - start
                try:
                    written = os.path.getsize(output_path)
                except OSError:
                    written = 0

                samples.append((time.time(), written))
                if len(samples) > _window:
                    samples.pop(0)

                mb_written = written / 1_048_576
                speed_str = ""
                if len(samples) >= 2 and written > 0:
                    dt = samples[-1][0] - samples[0][0]
                    db = samples[-1][1] - samples[0][1]
                    if dt > 0 and db > 0:
                        speed_str = f" | {(db / 1_048_576) / dt:.1f} MB/s"

                mb_str = f" | {mb_written:.1f} MB written" if written > 0 else ""
                line = (
                    f"\r  {spinner[i % len(spinner)]} "
                    f"{elapsed:.0f}s{mb_str}{speed_str}   "
                )
                print(line, end="", flush=True)
                time.sleep(0.4)
                i += 1

        t = threading.Thread(target=_ticker, daemon=True)
        t.start()
        try:
            self.conn.execute(
                f"COPY ({query}) TO '{output_path}' (HEADER, DELIMITER '|')"
            )
        finally:
            done.set()
            t.join()

        elapsed = time.time() - start
        try:
            final_mb = os.path.getsize(output_path) / 1_048_576
            size_str = f" | {final_mb:.1f} MB"
        except OSError:
            size_str = ""
        print(f"\r  done in {elapsed:.1f}s{size_str}                    ")

    def extract_all(
        self,
        metadata_path="src/data_processing/extraction/extraction_metadata.json",
        tables=None,
    ):
        """
        Extract all tables defined in the metadata JSON.

        Parameters
        ----------
        metadata_path : str
            Path to the extraction_metadata.json configuration file.
        tables : list of str, optional
            If provided, only extracts the named table keys.
        """
        with open(metadata_path) as f:
            metadata = json.load(f)

        items = [(k, v) for k, v in metadata.items() if not tables or k in tables]
        total = len(items)
        for idx, (key, conf) in enumerate(items, start=1):
            self.extract_table(key, conf, index=idx, total=total)

    def extract_table(self, name, conf, index=None, total=None):
        """
        Extract a single table to a pipe-delimited CSV file.

        Parameters
        ----------
        name : str
            Logical name of the table (used for logging).
        conf : dict
            Table configuration entry from extraction_metadata.json.
        index : int, optional
            1-based position of this table in the extraction sequence.
        total : int, optional
            Total number of tables being extracted in this run.
        """
        counter = f" [{index}/{total}]" if index is not None else ""
        print(f"\n--- Extracting {name}{counter} ---")
        output_csv = os.path.join(self.export_dir, f"{conf['file_name']}.csv")

        if os.path.exists(output_csv):
            print(f"  Skipping — output already exists: {output_csv}")
            return

        query = self._prepare_query(conf["query"])
        output_path = os.path.abspath(output_csv).replace("\\", "/")

        print("  Running query...")
        self._run_with_timer(query, output_path)
        print(f"  Saved to {output_csv}")

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()
