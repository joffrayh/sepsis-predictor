import argparse
import os
import psycopg2 as pg
import json

# define constants
DBNAME = 'mimiciv'
EXPORT_DIR = 'test'

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", help="MIMIC Database Username", type=str, required=True)
parser.add_argument("-p", "--password", help="MIMIC Database Password", type=str, default="")
pargs = parser.parse_args()

# initialise database connection
conn = pg.connect(F"dbname={DBNAME} user={pargs.username} host='localhost' options='--search_path=mimiciv' password={pargs.password}")

# create output dir
exportdir = os.path.join(os.getcwd(), EXPORT_DIR)
if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# TODO need to change from absolute path to use current path (os.path....)
# load metadata needed for extraction
with open("src/MIMIC-sepsis/src/extraction/extraction_metadata.json", "r") as f:
    extraction_metadata = json.load(f)

# iterate through each etraction step needed
with conn.cursor() as cur:
    for data, metadata in extraction_metadata.items(): 

        print(data)
        print(metadata)

        print(f"Extracting {metadata['extraction_text']}...")

        output_file = os.path.join(exportdir, metadata['file_name']+'.csv')

        with open(output_file, 'w') as f:
            cur.copy_expert(
                f"""
                COPY (
                    {metadata['query']}
                )
                TO STDOUT WITH CSV HEADER DELIMITER '|'
                """,
                f
            )

        print(f"Success! {metadata['extraction_text']} saved to: {output_file}")