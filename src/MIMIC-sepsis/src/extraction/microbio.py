import argparse
import os
import psycopg2 as pg

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", help="MIMIC Database Username", type=str, required=True)
parser.add_argument("-p", "--password", help="MIMIC Database Password", type=str, default="")
pargs = parser.parse_args()

# Initializing database connection
conn = pg.connect("dbname='mimiciv' user={0} host='localhost' options='--search_path=mimiciv' password={1}".format(pargs.username,pargs.password))

# create output dir
exportdir = os.path.join(os.getcwd(), 'processed_files')
if not os.path.exists(exportdir):
    os.makedirs(exportdir)

output_file = os.path.join(exportdir,'microbio.csv')


# 2. microbio (Microbiologyevents)
# extract(epoch from charttime) : The number of seconds since 1970-01-01 00:00:00 UTC

print("Extracting microbiology events data...")

with conn.cursor() as cur:
    with open(output_file, 'w') as f:
        cur.copy_expert(
            """
            COPY (
                select subject_id,
                    hadm_id, 
                    extract(epoch from charttime) as charttime, 
                    extract(epoch from chartdate) as chartdate 
                from mimiciv_hosp.microbiologyevents
            )
            TO STDOUT WITH CSV HEADER DELIMITER '|'
            """,
            f
        )

print(f"Success! Data saved to: {output_file}")