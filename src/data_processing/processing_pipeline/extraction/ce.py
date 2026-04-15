import argparse
import os
import math
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
import subprocess

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
    END AS valuenum
FROM mimiciv_icu.chartevents
{where_clause}
"""

# excecute the query in chunks and write to CSV
chunk_size = 100000
temp_file = os.path.join(exportdir, 'chartevents_temp.csv')
final_file = os.path.join(exportdir, 'chartevents.csv')

print(f"Target file: {final_file}")

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
            
            chunk.to_csv(temp_file, mode=mode, header=header, index=False, sep='|')
            
            first_chunk = False

    # sort and remove duplicates
    print("\nSorting and removing duplicates...")

    # "C" locale for speed
    sort_env = os.environ.copy()
    sort_env["LC_ALL"] = "C"
    
    # tell 'sort' to use our export directory for temp files
    # don't use /tmp as it has limited space
    sort_env["TMPDIR"] = exportdir 

    # write the header
    with open(temp_file, 'rb') as f_in:
        header = f_in.readline()
    with open(final_file, 'wb') as f_out:
        f_out.write(header)

    # The Command: Read from Stdin -> Sort -> Uniq -> Append to File
    cmd = f"sort -t '|' -k1,1 -k2,2n --parallel=4 | uniq >> {final_file}"

    # start the subprocess with the updated environment
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, env=sort_env)

    try:
        with open(temp_file, 'rb') as f_in:
            # skip the header in the input (we already wrote it)
            f_in.readline()
            
            # get total size for tqdm
            total_bytes = os.path.getsize(temp_file)
            
            # read in 64MB chunks
            chunk_size = 64 * 1024 * 1024 
            
            # feed the chunks to the sort process 
            with tqdm(total=total_bytes, unit='B', unit_scale=True, desc="Sorting") as pbar:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    try:
                        process.stdin.write(chunk)
                        pbar.update(len(chunk))
                    except BrokenPipeError:
                        # this catches the error if 'sort' dies (e.g. no space)
                        break
        
        # close stdin to signal to 'sort' that we are done sending data
        process.stdin.close()
        
        print("Data read complete. Waiting for final merge (this usually takes 1-2 mins)...")
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
    except Exception as e:
        process.kill()
        raise e

    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"Completed! Final sorted file saved at: {final_file}")
except Exception as e:
    print(f"\nError occurred: {e}")