import json
import os

import psycopg2 as pg


class MIMICExtractor:
    """
    Unified extractor for a MIMIC-IV PostgreSQL database.

    Driven by ``extraction_metadata.json`` to handle varying dataset sizes.
    Uses psycopg2 ``COPY`` for all bulk extraction directly to CSV.
    """

    def __init__(
        self,
        user,
        password="",
        host="localhost",
        port=5432,
        dbname="mimiciv",
        export_dir="processed_files",
    ):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.export_dir = os.path.join(os.getcwd(), export_dir)

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        search_path = "--search_path=mimiciv_hosp,mimiciv_icu,public"
        conn_string = (
            f"dbname='{self.dbname}' user='{self.user}' "
            f"host='{self.host}' port='{self.port}' "
            f"options='{search_path}'"
        )
        if self.password:
            conn_string += f" password='{self.password}'"
        self.pg_conn = pg.connect(conn_string)

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

    def _execute_direct_copy(self, conf, output_csv):
        """Fastest extraction method - writes direct from Postgres engine to disk."""
        query = conf["query"]
        with self.pg_conn.cursor() as cur, open(output_csv, "w") as f:
            cur.copy_expert(
                f"COPY ({query}) TO STDOUT WITH CSV HEADER DELIMITER '|'", f
            )

    def extract_table(self, name, conf):
        """Orchestrates individual table extraction based on the metadata rules."""
        print(f"\n--- Extracting {name} ---")
        output_csv = os.path.join(self.export_dir, f"{conf['file_name']}.csv")

        if os.path.exists(output_csv):
            print(f"Skipping {name}, target file already exists: {output_csv}")
            return

        # Standard extraction
        print(f"Executing standard COPY for {name}...")
        self._execute_direct_copy(conf, output_csv)
        print(f"Success! Saved to {output_csv}")

    def close(self):
        """Close all open database connections."""
        self.pg_conn.close()
