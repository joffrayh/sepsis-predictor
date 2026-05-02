import glob
import json
import os
import re
import threading
import time

import duckdb


class MIMICExtractor:
    """
    Stateful MIMIC-IV extractor using an in-process DuckDB connection.

    On construction, registers every ``.csv.gz`` file under ``hosp/`` and
    ``icu/`` as a DuckDB view so that the SQL in
    ``extraction_metadata.json`` can reference tables by schema-qualified
    name without a running PostgreSQL server. The
    connection persists across all ``extract_table`` calls and must be released
    via ``close()`` when extraction is complete.
    """

    def __init__(
        self,
        raw_data_dir,
        export_dir,
    ):
        """
        Open a DuckDB connection and register all MIMIC-IV tables as views.

        Parameters
        ----------
        raw_data_dir : str
            Path to the top-level MIMIC-IV directory. Must contain ``hosp/`` and
            ``icu/`` subdirectories of ``.csv.gz`` files.
        export_dir : str
            Destination directory for extracted pipe-delimited CSVs. Created if
            absent. Resolved to an absolute path at construction time —
            pass an absolute path if the working directory may change
            between construction and extraction.

        Notes
        -----
        The DuckDB connection is in-process (no socket, no server, no
        credentials). ``close()`` must be called when extraction is
        complete to release file handles and DuckDB resources;
        ``main.py`` ensures this in a ``finally`` block.
        """
        self.raw_data_dir = raw_data_dir
        self.export_dir = os.path.join(os.getcwd(), export_dir)
        os.makedirs(self.export_dir, exist_ok=True)

        self.conn = duckdb.connect()
        self._register_views()

    def _register_views(self):
        """
        Register MIMIC-IV ``.csv.gz`` files as DuckDB views under their schema.

        Files under ``hosp/`` become ``mimiciv_hosp.<stem>`` views; files under
        ``icu/`` become ``mimiciv_icu.<stem>`` views, where ``<stem>`` is the
        filename with ``.csv.gz`` stripped. Every file present is registered
        regardless of whether it appears in ``extraction_metadata.json`` —
        unused views are harmless.

        Notes
        -----
        File paths are normalised to forward slashes before embedding in SQL to
        ensure DuckDB resolves them correctly on both POSIX and Windows systems.
        """
        schema_subdirs = {
            "mimiciv_hosp": os.path.join(self.raw_data_dir, "hosp"),
            "mimiciv_icu": os.path.join(self.raw_data_dir, "icu"),
        }
        for schema, directory in schema_subdirs.items():
            self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            for gz_path in glob.glob(os.path.join(directory, "*.csv.gz")):
                table_name = os.path.basename(gz_path).replace(".csv.gz", "")
                # DuckDB requires forward slashes in embedded SQL path strings
                abs_path = os.path.abspath(gz_path).replace("\\", "/")
                self.conn.execute(
                    f"CREATE VIEW {schema}.{table_name} AS "
                    f"SELECT * FROM read_csv_auto('{abs_path}')"
                )
                print(f"  Registered view: {schema}.{table_name}")

    def _prepare_query(self, query):
        """
        Normalise a query for DuckDB execution.

        Strip the PostgreSQL-specific ``CREATE TEMP TABLE ... AS`` prefix and
        trailing semicolons from a raw SQL string, yielding a plain SELECT or
        CTE that can be embedded in a DuckDB ``COPY (...) TO ...`` statement.

        Parameters
        ----------
        query : str
            Raw SQL string from ``extraction_metadata.json``.

        Returns
        -------
        str
            Clean SELECT or CTE string with no leading DDL and no trailing
            semicolon.

        Notes
        -----
        The ``CREATE TEMP TABLE`` stripping exists solely to handle the one
        legacy PostgreSQL-style query in the metadata (``demog``). The regex
        is a no-op on all other queries.
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
        Execute a DuckDB COPY query with live elapsed-time and speed display.

        A daemon ticker thread wakes every 0.4 s, samples the output file size,
        and computes a rolling MB/s estimate over a 5-sample window. The actual
        export runs in the calling thread; the ticker is joined before
        the method returns even if the ``COPY`` raises.

        Parameters
        ----------
        query : str
            Prepared SELECT query (output of ``_prepare_query``).
        output_path : str
            Absolute forward-slash path for the output CSV.

        Notes
        -----
        Output is always written as a pipe-delimited CSV with a header row
        (``DELIMITER '|'``, ``HEADER``). All downstream readers must use
        ``sep="|"``.

        The ticker thread is a daemon and is unconditionally joined in a
        ``finally`` block, so the progress display is always cleaned up
        regardless of whether the export succeeds or fails.
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
                # Suppress speed until at least two samples with non-zero bytes
                # exist — avoids division-by-zero on fast queries that
                # buffer before flush
                if len(samples) >= 2 and written > 0:
                    dt = samples[-1][0] - samples[0][0]
                    db = samples[-1][1] - samples[0][1]
                    if dt > 0 and db > 0:
                        speed_str = f" | {(db / 1_048_576) / dt:.1f} MB/s"

                mb_str = (
                    f" | {mb_written:.1f} MB written" if written > 0 else ""
                )
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
        metadata_path,
        tables=None,
    ):
        """
        Extract all tables defined in the metadata JSON.

        Parameters
        ----------
        metadata_path : str
            Path to ``extraction_metadata.json``.
        tables : list of str, optional
            Subset of table keys to extract. If ``None``, all entries are
            extracted.

        Notes
        -----
        Extraction order follows JSON insertion order (Python 3.7+ dict
        ordering). Tables are independent — the order has no semantic
        significance and does not affect downstream phases.
        """
        with open(metadata_path) as f:
            metadata = json.load(f)

        items = [
            (k, v) for k, v in metadata.items() if not tables or k in tables
        ]
        total = len(items)
        for idx, (key, conf) in enumerate(items, start=1):
            self.extract_table(key, conf, index=idx, total=total)

    def extract_table(self, name, conf, index=None, total=None):
        """
        Extract a single table to a pipe-delimited CSV.

        Parameters
        ----------
        name : str
            Logical table name used for progress logging.
        conf : dict
            Entry from ``extraction_metadata.json``. Must contain ``"query"``
            (raw SQL) and ``"file_name"`` (output stem without ``.csv``).
        index : int, optional
            1-based position of this table in the current extraction run,
            used for the ``[N/M]`` log prefix.
        total : int, optional
            Total number of tables in this run.

        Notes
        -----
        Skips silently if the output CSV already exists, providing a per-table
        checkpoint for interrupted runs. A partially written file from a
        previous crash would pass this check and be treated as complete —
        delete the suspect CSV manually before re-running if results look wrong.
        """
        counter = f" [{index}/{total}]" if index is not None else ""
        print(f"\n--- Extracting {name}{counter} ---")
        output_csv = os.path.join(self.export_dir, f"{conf['file_name']}.csv")

        if os.path.exists(output_csv):
            print(f"  Skipping — output already exists: {output_csv}")
            return

        query = self._prepare_query(conf["query"])
        # DuckDB requires forward slashes in embedded SQL path strings
        output_path = os.path.abspath(output_csv).replace("\\", "/")

        print("  Running query...")
        self._run_with_timer(query, output_path)
        print(f"  Saved to {output_csv}")

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()
