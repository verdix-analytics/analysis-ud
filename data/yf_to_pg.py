import os
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values
import yfinance as yf

# US market symbols
SYMBOLS = [
    "AAPL", "GOOGL", "TSLA", "MSFT", "NVDA", "META", "AMZN", "AVGO", "ORCL", "AMD",
    "INTC", "QCOM", "TXN", "CRM", "ADBE", "NFLX", "UBER", "NOW", "SNOW", "PLTR", "JPM",
    "BAC", "WFC", "GS", "MS", "BLK", "AXP", "SPGI", "V", "MA", "BRK-B", "C", "SCHW",
    "COF", "USB", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "MDT", "ISRG",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "WMT", "COST", "TGT", "MCD",
    "SBUX", "NKE", "HD", "LOW", "PG", "KO", "CAT", "HON", "GE", "BA", "RTX", "LMT", "UPS",
    "DE", "DIS", "CMCSA", "T", "VZ", "TMUS", "SPY", "QQQ", "IWM", "DIA",
]

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "020101")
PG_DB = os.getenv("PG_DB", "zyx")

# Cloud DB (Aiven Postgres) connection settings
#CLOUD_DB_URL = os.getenv(
   # "CLOUD_DB_URL",
   # "postgres://avnadmin:AVNS_nGhKkso1lhBUZQ5Oa9N@capstone-cs50-1fd3bfbe-capstone-cs50-group2.l.aivencloud.com:21430/defaultdb?sslmode=require",
#)
#CLOUD_DB_NAME = os.getenv("CLOUD_DB_NAME", "defaultdb")
#CLOUD_DB_HOST = os.getenv("CLOUD_DB_HOST", "capstone-cs50-1fd3bfbe-capstone-cs50-group2.l.aivencloud.com")
#CLOUD_DB_PORT = int(os.getenv("CLOUD_DB_PORT", "21430"))
#CLOUD_DB_PASSWORD = os.getenv("CLOUD_DB_PASSWORD", "AVNS_nGhKkso1lhBUZQ5Oa9N")
TABLE_NAME = "finance_prices"
VIEW_NAME  = "ohlcv_1h"

def get_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )


def ensure_timescaledb_extension(conn):
    """Ensure timescaledb extension exists."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    conn.commit()


def ensure_table(conn):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        symbol TEXT NOT NULL,
        trade_date TIMESTAMPTZ NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        adj_close NUMERIC,
        volume BIGINT,
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (symbol, updated_at)
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()

def ensure_1h_view(conn):
    ddl = f"""
    CREATE MATERIALIZED VIEW IF NOT EXISTS {VIEW_NAME} AS
    WITH
      opens AS (
        SELECT DISTINCT ON (symbol, date_trunc('hour', trade_date))
               symbol,
               date_trunc('hour', trade_date) AS bucket,
               open
        FROM {TABLE_NAME}
        ORDER BY symbol, date_trunc('hour', trade_date), trade_date ASC
      ),
      closes AS (
        SELECT DISTINCT ON (symbol, date_trunc('hour', trade_date))
               symbol,
               date_trunc('hour', trade_date) AS bucket,
               close,
               adj_close
        FROM {TABLE_NAME}
        ORDER BY symbol, date_trunc('hour', trade_date), trade_date DESC
      ),
      base AS (
        SELECT
          symbol,
          date_trunc('hour', trade_date) AS bucket,
          MAX(high)      AS high,
          MIN(low)       AS low,
          SUM(volume)    AS volume,
          MAX(updated_at) AS updated_at
        FROM {TABLE_NAME}
        GROUP BY symbol, date_trunc('hour', trade_date)
      )
    SELECT
      b.symbol,
      b.bucket       AS trade_date,
      o.open,
      b.high,
      b.low,
      c.close,
      c.adj_close,
      b.volume,
      b.updated_at
    FROM base b
    JOIN opens  o USING (symbol, bucket)
    JOIN closes c USING (symbol, bucket)
    WITH NO DATA;

    -- indexes for fast queries
    CREATE UNIQUE INDEX IF NOT EXISTS {VIEW_NAME}_symbol_date_idx
        ON {VIEW_NAME} (symbol, trade_date DESC);

    CREATE INDEX IF NOT EXISTS {VIEW_NAME}_date_idx
        ON {VIEW_NAME} (trade_date DESC);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def refresh_1h_view(conn):
    """Call this on a schedule (e.g. every hour) to update the view."""
    with conn.cursor() as cur:
        cur.execute(f"REFRESH MATERIALIZED VIEW {VIEW_NAME};")
    conn.commit()

def upsert_rows(conn, rows):
    if not rows:
        return 0
    sql = f"""
        INSERT INTO {TABLE_NAME} (symbol, trade_date, open, high, low, close, adj_close, volume, updated_at)
        VALUES %s;
        
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def download_daily(symbols):
    df = yf.download(
        symbols,
        period="8d",
        interval="1m",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )
    return df


def build_rows(df, symbols):
    rows = []
    if df.empty:
        return rows

    for symbol in symbols:
        if symbol not in df.columns.get_level_values(0):
            continue
        sym_df = df[symbol].dropna(how="all")
        for idx, row in sym_df.iterrows():
            # keep minute-level timestamp (not just date)
            trade_date = idx.to_pydatetime()
            rows.append(
                (
                    symbol,
                    trade_date,
                    None if row.get("Open") is None else float(row["Open"]),
                    None if row.get("High") is None else float(row["High"]),
                    None if row.get("Low") is None else float(row["Low"]),
                    None if row.get("Close") is None else float(row["Close"]),
                    None if row.get("Adj Close") is None else float(row["Adj Close"]),
                    None if row.get("Volume") is None else int(row["Volume"]),
                    datetime.utcnow(),
                )
            )
    return rows


def main():
    print(f"Downloading {len(SYMBOLS)} symbols...")
    df = download_daily(SYMBOLS)
    print("Download complete. Preparing rows...")
    rows = build_rows(df, SYMBOLS)
    print(f"Prepared {len(rows)} rows.")

    conn = get_conn()
    try:
        ensure_timescaledb_extension(conn)
        ensure_table(conn)
        ensure_1h_view(conn)
        written = upsert_rows(conn, rows)
        # Initial populate
        refresh_1h_view(conn)
        print(f"Upserted {written} rows into {TABLE_NAME}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
