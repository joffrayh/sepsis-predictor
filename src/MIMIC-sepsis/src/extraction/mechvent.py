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

output_file = os.path.join(exportdir,'mechvent.csv')

# mechvent (Mechanical ventilation)
print("Extracting mechanical ventilation data...")

with conn.cursor() as cur:
    with open(output_file, 'w') as f:
        cur.copy_expert(
            """
            COPY (
                select
                    stay_id, extract(epoch from charttime) as charttime,   
                    -- case statement determining whether it is an instance of mech vent
                    max(        
                        case
                            when itemid is null or value is null then 0 -- can't have null values
                            when itemid = 223894 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
                            when itemid = 226732 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
                            when itemid in (
                                224687 -- minute volume
                                , 224685, 224684, 224686 -- tidal volume
                                , 224697, 224695, 224696, 224746, 224747 
                                , 226873, 224738, 224419, 224750, 227187 -- Insp pressure
                                , 224707, 224709, 224705, 224706 -- APRV pressure
                                , 220339, 224700 -- PEEP
                                , 224702 -- PCV
                                , 227809, 227810 -- ETT
                                , 224701 -- PSVlevel
                            ) THEN 1
                            else 0
                        end
                    ) as mechvent
                from mimiciv_icu.chartevents ce
                where 
                    value is not null and
                    itemid in (
                        223894 -- vent type
                        , 226732 -- O2 delivery device
                        , 224687 -- minute volume
                        , 224685, 224684, 224686 -- tidal volume
                        , 224697, 224695, 224696, 224746, 224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
                        , 226873, 224738, 224419, 224750, 227187 -- Insp pressure
                        , 224707, 224709, 224705, 224706 -- APRV pressure
                        , 220339, 224700 -- PEEP
                        , 224702 -- PCV
                        , 227809, 227810 -- ETT
                        , 224701 -- PSVlevel
                    )
                group by stay_id, charttime            
            )
            TO STDOUT WITH CSV HEADER DELIMITER '|'
            """,
            f
        )

print(f"Success! Data saved to: {output_file}")