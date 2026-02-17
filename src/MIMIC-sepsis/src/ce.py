import argparse
import os
import math
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--username", help="MIMIC Database Username", type=str, required=True)
parser.add_argument("-p", "--password", help="MIMIC Database Password", type=str, default="")
pargs = parser.parse_args()

# connect to database via alchemy engine
print(f"Connecting to database 'mimiciv' as user '{pargs.username}'...")
auth_string = f"{pargs.username}"
if pargs.password:
    auth_string += f":{pargs.password}"

db_url = f"postgresql+psycopg2://{auth_string}@localhost/mimiciv"
engine = create_engine(db_url, connect_args={'options': '-c search_path=mimiciv_hosp,mimiciv_icu,public'})


# create output dir
exportdir = os.path.join(os.getcwd(), 'processed_files')
if not os.path.exists(exportdir):
    os.makedirs(exportdir)

# define filter condition (used for both Count and Select)
# we separate this so we ensure both queries look for the exact same data
where_clause = """
WHERE valuenum IS NOT NULL
  OR (itemid IN (223834, 226732, 228096) AND value IS NOT NULL)
AND itemid IN (226732, 223834, 227287, 224691, 226707, 226730, 224639, 226512, 226531, 228096,
               220045, 220179, 225309, 220050, 227243, 224167, 220181, 220052, 225312, 224322,
               225310, 224643, 227242, 220051, 220180, 220210, 224422, 224690, 220277, 220227,
               223762, 223761, 224027, 220074, 228368, 228177, 223835, 220339, 
               224700, 224686, 224684, 224421, 224687, 224697, 224695, 224696)
"""

# main query to extract chartevents data 
# with mappings for Oxygen Flow Device and RASS score
select_query = f"""
SELECT stay_id, 
    extract(epoch from charttime) as charttime, 
    itemid,
    CASE
        -- Oxygen Flow Device mapping
        WHEN itemid = 223834 AND value = 'None' THEN '0'
        WHEN itemid = 223834 AND value = 'Nasal cannula' THEN '2'
        WHEN itemid = 223834 AND value = 'Face tent' THEN '3'
        WHEN itemid = 223834 AND value = 'Aerosol-cool' THEN '4'
        WHEN itemid = 223834 AND value = 'Trach mask' THEN '5'
        WHEN itemid = 223834 AND value = 'High flow nasal cannula' THEN '6'
        WHEN itemid = 223834 AND value = 'High flow neb' THEN '6'
        WHEN itemid = 223834 AND value = 'Non-rebreather' THEN '7'
        WHEN itemid = 223834 AND value = 'Venti mask' THEN '8'
        WHEN itemid = 223834 AND value = 'Medium conc mask' THEN '9'
        WHEN itemid = 223834 AND value = 'Endotracheal tube' THEN '10'
        WHEN itemid = 223834 AND value = 'Tracheostomy tube' THEN '11'
        WHEN itemid = 223834 AND value = 'T-piece' THEN '12'
        WHEN itemid = 223834 AND value = 'CPAP mask' THEN '13'
        WHEN itemid = 223834 AND value = 'Bipap mask' THEN '13'
        WHEN itemid = 223834 AND value = 'Oxymizer' THEN '14'
        WHEN itemid = 223834 AND value = 'Other' THEN '15'

        -- Oxygen Flow Device mapping for item 226732
        WHEN itemid = 226732 AND value = 'None' THEN '0'
        WHEN itemid = 226732 AND value = 'Nasal cannula' THEN '2'
        WHEN itemid = 226732 AND value = 'Face tent' THEN '3'
        WHEN itemid = 226732 AND value = 'Aerosol-cool' THEN '4'
        WHEN itemid = 226732 AND value = 'Trach mask' THEN '5'
        WHEN itemid = 226732 AND value = 'High flow nasal cannula' THEN '6'
        WHEN itemid = 226732 AND value = 'High flow neb' THEN '6'
        WHEN itemid = 226732 AND value = 'Non-rebreather' THEN '7'
        WHEN itemid = 226732 AND value = 'Venti mask' THEN '8'
        WHEN itemid = 226732 AND value = 'Medium conc mask' THEN '9'
        WHEN itemid = 226732 AND value = 'Endotracheal tube' THEN '10'
        WHEN itemid = 226732 AND value = 'Tracheostomy tube' THEN '11'
        WHEN itemid = 226732 AND value = 'T-piece' THEN '12'
        WHEN itemid = 226732 AND value = 'CPAP mask' THEN '13'
        WHEN itemid = 226732 AND value = 'Bipap mask' THEN '13'
        WHEN itemid = 226732 AND value = 'Oxymizer' THEN '14'
        WHEN itemid = 226732 AND value = 'Ultrasonic neb' THEN '15'
        WHEN itemid = 226732 AND value = 'Vapomist' THEN '16'
        WHEN itemid = 226732 AND value = 'Other' THEN '17'

        -- RASS score mapping
        WHEN itemid = 228096 AND value LIKE '0%Alert and calm%' THEN '0'
        WHEN itemid = 228096 AND value LIKE '-1%Awakens to voice%' THEN '-1'
        WHEN itemid = 228096 AND value LIKE '-2%Light sedation%' THEN '-2'
        WHEN itemid = 228096 AND value LIKE '-3%Moderate sedation%' THEN '-3'
        WHEN itemid = 228096 AND value LIKE '-4%Deep sedation%' THEN '-4'
        WHEN itemid = 228096 AND value LIKE '-5%Unarousable%' THEN '-5'
        WHEN itemid = 228096 AND value LIKE '+1%Anxious%' THEN '1'
        WHEN itemid = 228096 AND value LIKE '+2%Frequent nonpurposeful%' THEN '2'
        WHEN itemid = 228096 AND value LIKE '+3%Pulls or removes tube%' THEN '3'
        WHEN itemid = 228096 AND value LIKE '+4%Combative%' THEN '4'
        
        ELSE CAST(valuenum AS TEXT)
    END AS value
FROM mimiciv_icu.chartevents
"""

# excecute the query in chunks and write to CSV
chunk_size = 100000
output_file = os.path.join(exportdir, 'chartevents.csv')

print(f"Target file: {output_file}")

try:
    # we set stream_results=True to enable server-side cursor and 
    # avoid loading all data into memory
    with engine.connect().execution_options(stream_results=True) as conn:
        
        # count rows first, to get estimate of processing time
        print("Calculating total rows (this may take a few minutes)...")
        count_query = f"SELECT count(*) FROM mimiciv_icu.chartevents {where_clause}"
        total_rows = conn.execute(text(count_query)).scalar()
        
        total_chunks = math.ceil(total_rows / chunk_size)
        print(f"Total rows to process: {total_rows:,} (approx {total_chunks} chunks)")

        # execute the main query in chunks and write to CSV
        print("Starting extraction...")
        chunks = pd.read_sql_query(text(select_query), conn, chunksize=chunk_size)
        
        first_chunk = True
        
        # iterate over chunks and write to CSV
        for chunk in tqdm(chunks, total=total_chunks, desc="Progress", unit="chunk"):
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            
            chunk.to_csv(output_file, mode=mode, header=header, index=False, sep='|')
            
            first_chunk = False

    print(f"\nCompleted!")

except Exception as e:
    print(f"\nError occurred: {e}")