import os
import json
import psycopg2 as pg
from sqlalchemy import create_engine


class MIMICExtractor:
    """
    A unified class to extract data from a MIMIC-IV PostgreSQL database.
    Driven by extraction_metadata.json parameters to handle varying dataset sizes.
    """

    def __init__(
        self,
        user,
        password="",
        host="localhost",
        dbname="mimiciv",
        export_dir="processed_files",
    ):
        self.user = user
        self.password = password
        self.host = host
        self.dbname = dbname
        self.export_dir = os.path.join(os.getcwd(), export_dir)

        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

        # Connect strictly with Psycopg2 for fast COPY commands
        conn_string = f"dbname='{self.dbname}' user='{self.user}' host='{self.host}' options='--search_path=mimiciv,mimiciv_icu,mimiciv_hosp,public'"
        if self.password:
            conn_string += f" password='{self.password}'"
        self.pg_conn = pg.connect(conn_string)

        # Connect using SQLAlchemy for chunked Pandas reads when sorting is required
        auth_string = f"{self.user}"
        if self.password:
            auth_string += f":{self.password}"
        db_url = f"postgresql+psycopg2://{auth_string}@{self.host}/{self.dbname}"
        self.alchemy_engine = create_engine(
            db_url,
            connect_args={"options": "-c search_path=mimiciv_hosp,mimiciv_icu,public"},
        )

    def extract_all(
        self,
        metadata_path="src/data_processing/processing_pipeline/extraction/extraction_metadata.json",
        tables=None,
    ):
        """
        Reads the metadata JSON and extracts every target.
        If `tables` is provided, only extracts those specifically named.
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        for key, conf in metadata.items():
            if tables and key not in tables:
                continue

            self.extract_table(key, conf)

    def _execute_direct_copy(self, conf, output_csv):
        """Fastest extraction method - writes direct from Postgres engine to disk."""
        query = conf["query"]
        with self.pg_conn.cursor() as cur:
            with open(output_csv, "w") as f:
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
        self.pg_conn.close()
        self.alchemy_engine.dispose()
