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

output_file = os.path.join(exportdir,'uo.csv')

# 8. uo (Real-time Urine Output)
"""
 Itemid | Label
-----------------------------------------------------
 226559 | Foley
 226560 | Void
 227510 | TF Residual
 226561 | Condom Cath
 227489 | GU Irrigant/Urine Volume Out
 226584 | Ileoconduit
 226563 | Suprapubic
 226564 | R Nephrostomy
 226565 | L Nephrostomy
 226557 | R Ureteral Stent
 226558 | L Ureteral Stent
 226713 | Incontinent/voids (estimate)
 226567 | Straight Cath
"""


# mechvent (Mechanical ventilation)
print("Extracting urine output data...")

with conn.cursor() as cur:
    with open(output_file, 'w') as f:
        cur.copy_expert(
            """
            COPY (
                select stay_id, 
                    extract(epoch from charttime) as charttime, 
                    itemid, 
                    value
                from mimiciv_icu.outputevents
                where stay_id is not null and value is not null and itemid in (
                    226559, 226560, 227510, 226561, 227489,
                    226584, 226563, 226564, 226565, 226557, 226558, 226713, 226567
                )
                order by stay_id, charttime, itemid
            )
            TO STDOUT WITH CSV HEADER DELIMITER '|'
            """,
            f
        )

print(f"Success! Data saved to: {output_file}")