#!/usr/bin/env bash
# Exit on error (-e), treat unset vars as errors (-u), 
# and fail on pipeline errors (-o pipefail)
set -euo pipefail

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

## Resolve script directory and default data location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_DIR="$SCRIPT_DIR/../../../data/raw/mimic-iv-3.1"

## Inputs: optional data directory arg and DB connection env vars
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
DB_NAME="${DB_NAME:-mimiciv}"
DB_USER="${DB_USER:-${USER}}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

## Where we keep the official MIMIC-IV Postgres schema scripts
SQL_DIR="$SCRIPT_DIR/sql"
CREATE_SQL="$SQL_DIR/create.sql"

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------

## Ensure the SQL script directory exists
mkdir -p "$SQL_DIR"

## Validate data directory
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Error: Data directory not found: $DATA_DIR" >&2
  echo "Usage: $0 /absolute/path/to/mimic-iv-3.1" >&2
  exit 1
fi

## Ensure expected MIMIC-IV folder layout
if [[ ! -d "$DATA_DIR/hosp" || ! -d "$DATA_DIR/icu" ]]; then
  echo "Error: Expected subfolders not found under $DATA_DIR (hosp/, icu/)." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# PREPARATION
# -----------------------------------------------------------------------------

## Download the official schema creation SQL if missing
if [[ ! -f "$CREATE_SQL" ]]; then
  echo "Downloading MIMIC-IV Postgres schema script (create.sql)..."
  if command -v curl >/dev/null 2>&1; then
    curl -L "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/buildmimic/postgres/create.sql" -o "$CREATE_SQL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$CREATE_SQL" "https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/buildmimic/postgres/create.sql"
  else
    echo "Error: Neither curl nor wget found; please install one to download schema SQL." >&2
    exit 1
  fi
fi

## Check if the target database exists, create it if needed
echo "Checking database: $DB_NAME"
DB_EXISTS=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'")
if [[ "$DB_EXISTS" != "1" ]]; then
  echo "Creating database $DB_NAME..."
  createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
fi

## Create schemas/tables (drops mimiciv_hosp/mimiciv_icu if they exist)
echo "Creating schemas/tables (this will DROP existing MIMIC-IV schemas if present)..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$CREATE_SQL"

# -----------------------------------------------------------------------------
# DATA LOADING (STREAMING)
# -----------------------------------------------------------------------------

echo "Starting data load (Streaming directly from .gz files)..."

# Function to stream load a directory of files into a specific schema
load_schema() {
    local subdir=$1
    local schema=$2
    
    echo "Processing $subdir module..."

    # Check if files exist
    if ! compgen -G "$DATA_DIR/$subdir/*.csv.gz" > /dev/null; then
        echo "Warning: No .csv.gz files found in $DATA_DIR/$subdir"
        return
    fi

    for gzfile in "$DATA_DIR/$subdir"/*.csv.gz; do
        # Extract table name from filename (e.g., 'admissions.csv.gz' -> 'admissions')
        filename=$(basename "$gzfile")
        tablename="${filename%%.*}"
        
        # specific fix for chartevents chunks if they exist (usually not in 3.1, but good practice)
        # In MIMIC-IV v3.1 standard distribution, files map 1:1 to tables usually.
        
        full_table="$schema.$tablename"

        echo "  -> Loading table: $full_table (from $filename)"
        
        # Stream uncompressed data directly into Postgres
        # We use 'zcat' (or 'gzip -dc') to pipe to stdout
        # We use \COPY ... FROM STDIN to read from that pipe
        zcat "$gzfile" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -c "\COPY $full_table FROM STDIN WITH CSV HEADER NULL ''"
    done
}

# Load the two main modules
load_schema "hosp" "mimiciv_hosp"
load_schema "icu" "mimiciv_icu"

echo "-----------------------------------------"
echo "Success! Database $DB_NAME is ready."
echo "-----------------------------------------"