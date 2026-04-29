import glob
import json
import os
import re

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

    def extract_all(
        self,
        metadata_path="src/data_processing/processing_pipeline/extraction/extraction_metadata.json",
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

        for key, conf in metadata.items():
            if tables and key not in tables:
                continue
            self.extract_table(key, conf)

    def extract_table(self, name, conf):
        """
        Extract a single table to a pipe-delimited CSV file.

        Parameters
        ----------
        name : str
            Logical name of the table (used for logging).
        conf : dict
            Table configuration entry from extraction_metadata.json.
        """
        print(f"\n--- Extracting {name} ---")
        output_csv = os.path.join(self.export_dir, f"{conf['file_name']}.csv")

        if os.path.exists(output_csv):
            print(f"Skipping {name}, target file already exists: {output_csv}")
            return

        query = self._prepare_query(conf["query"])
        output_path = os.path.abspath(output_csv).replace("\\", "/")

        print(f"Executing DuckDB query for {name}...")
        self.conn.execute(f"COPY ({query}) TO '{output_path}' (HEADER, DELIMITER '|')")
        print(f"Success! Saved to {output_csv}")

    def close(self):
        """Close the DuckDB connection."""
        self.conn.close()
