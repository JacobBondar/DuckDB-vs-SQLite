import duckdb
import sqlite3
import os
import shutil
import time
import statistics
import platform
import psutil
import sqlglot
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Names for TPCH_TABLES
TPCH_TABLES = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp', 'region', 'supplier']

# Queries to skip for SQLite (only run on DuckDB) - per project requirements
SQLITE_SKIP_QUERIES = {7, 8, 9, 13}

# Number of SF rounds to compare results between DuckDB and SQLite
COMPARE_RESULTS_ROUNDS = 4

# Track which queries to skip per configuration (due to >90s timeout)
skip_queries = {
    "DuckDB": set(),
    "SQLite_No_Index": set(),
    "SQLite_With_Index": set()
}


# Module-level log file
_log_file = None

def open_log(filename):
    """Open log file for writing"""
    global _log_file
    _log_file = open(filename, "w", encoding="utf-8")

def close_log():
    """Close log file"""
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None

def log_print(*args, **kwargs):
    """Print to both console and log file"""
    global _log_file
    print(*args, **kwargs)
    if _log_file:
        print(*args, **kwargs, file=_log_file, flush=True)

def get_system_info():
    """
    Returns the info about the computer.
    """
    info = {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "RAM": f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    }
    log_print("=" * 40)
    log_print("System Information:")
    for k, v in info.items():
        log_print(f"  {k}: {v}")
    log_print("=" * 40)
    return info

def generate_datasets(sf):
    """
    Create data in DuckDB and then copies it to SQLite
    """
    log_print(f"\n--- Generating Data for SF = {sf} ---")

    clear_old_files()

    # Create DuckDB and the data
    con_duck = duckdb.connect("db_duck.db")
    con_duck.execute("INSTALL tpch; LOAD tpch;")
    con_duck.execute(f"CALL dbgen(sf={sf});")
    log_print(f"DuckDB data generated with sf={sf}!")

    # Create SQLite and data transport
    # Using Pandas to transport the data from DuckDB to SQLite
    con_sqlite = sqlite3.connect("db_sqlite_no_index.db")

    for table in TPCH_TABLES:
        log_print(f"Transferring table: {table}...")
        # Creating Data Frame from DuckDB
        df = con_duck.execute(f"SELECT * FROM {table}").df()
        # Transfer to SQLite
        df.to_sql(table, con_sqlite, index=False)

    log_print("  Data transferred to SQLite (No Index).")
    con_sqlite.close()
    con_duck.close()

    # Copy for indexed version
    shutil.copy("db_sqlite_no_index.db", "db_sqlite_with_index.db")
    log_print("  Created SQLite copy for indexing.")

def clear_old_files():
    """
    Clear old files
    :return: None
    """
    if os.path.exists("db_duck.db"):
        os.remove("db_duck.db")
    if os.path.exists("db_sqlite_no_index.db"):
        os.remove("db_sqlite_no_index.db")
    if os.path.exists("db_sqlite_with_index.db"):
        os.remove("db_sqlite_with_index.db")

def create_sqlite_indexes(con):
    """
    Creates indexes in columns for keys and dates.
    """
    log_print("Creating Indexes for SQLite...")
    cursor = con.cursor()

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(o_orderdate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_shipdate ON lineitem(l_shipdate)")

    con.commit()
    log_print("Indexes created successfully.")

def measure_query_time(con, query, is_duckdb=True):
    """
    Executes the query, measures time, then returns (median_time, result_df).
    If run takes more than 10 sec, runs only once (no median needed).
    Returns (None, None) if query fails.
    """
    times = []
    result_df = None

    # First run - also capture results
    start_time = time.time()
    try:
        if is_duckdb:
            result = con.execute(query).fetchdf()
        else:
            cursor = con.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            result = pd.DataFrame(rows, columns=columns)
        result_df = result
    except Exception as e:
        log_print(f"Query Failed: {e}")
        return None, None

    end_time = time.time()
    duration = end_time - start_time
    times.append(duration)

    # If execution takes more than 10 sec, run only once
    if duration > 10:
        return duration, result_df

    # Run 2 more times for median calculation (no need to capture results again)
    for i in range(2):
        start_time = time.time()
        try:
            if is_duckdb:
                con.execute(query).fetchall()
            else:
                cursor = con.cursor()
                cursor.execute(query)
                cursor.fetchall()
        except Exception as e:
            log_print(f"Query Failed on run {i+2}: {e}")
            return None, None
        end_time = time.time()
        times.append(end_time - start_time)

    return statistics.median(times), result_df

def get_tpch_query(query_number):
    """
    Gets the TPC-H query, from DuckDB. So we won't need to find and copy them all from TPC-H.
    """
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")

    # Get from the function tpch_queries() a virtual table, that has rows for every query (1-22)
    sql = f"SELECT query FROM tpch_queries() WHERE query_nr={query_number}"
    query = con.execute(sql).fetchone()[0]

    con.close()
    return query

def run_benchmark():
    """Main benchmark function"""
    open_log("output.txt")

    log_print("=" * 60)
    log_print("TPC-H Benchmark: DuckDB vs SQLite")
    log_print("=" * 60)

    get_system_info()

    scale_factors = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03]
    results_data = []

    for sf_idx, sf in enumerate(scale_factors):
        run_scales(sf, sf_idx, results_data)

    log_print("\n" + "=" * 60)
    log_print("  Benchmark Completed!")
    log_print("=" * 60)

    df_results = pd.DataFrame(results_data)
    df_results.to_csv("results.csv", index=False)
    log_print("Results saved to results.csv")
    generate_plots()

    close_log()
    print("Output saved to output.txt")

def run_scales(sf, sf_idx, results_data):
    log_print(f"\n{'='*50}")
    log_print(f"  Starting Scale Factor: {sf} (Round {sf_idx + 1})")
    log_print(f"{'='*50}")

    try:
        generate_datasets(sf)
    except Exception as e:
        log_print(f"Failed to generate data for SF = {sf}: {e}")
        return

    size_duck = os.path.getsize("db_duck.db") if os.path.exists("db_duck.db") else 0
    size_sqlite_no = os.path.getsize("db_sqlite_no_index.db") if os.path.exists("db_sqlite_no_index.db") else 0
    size_sqlite_idx = 0

    con_duck = duckdb.connect("db_duck.db")
    con_sqlite_no = sqlite3.connect("db_sqlite_no_index.db") if os.path.exists("db_sqlite_no_index.db") else None
    con_sqlite_idx = None

    if os.path.exists("db_sqlite_with_index.db"):
        con_sqlite_idx = sqlite3.connect("db_sqlite_with_index.db")
        create_sqlite_indexes(con_sqlite_idx)
        size_sqlite_idx = os.path.getsize("db_sqlite_with_index.db")

    for q_num in range(1, 23):
        run_queries(size_duck, size_sqlite_no, size_sqlite_idx, sf, sf_idx, q_num, results_data,
                    con_duck, con_sqlite_no, con_sqlite_idx)

    con_duck.close()
    if con_sqlite_no:
        con_sqlite_no.close()
    if con_sqlite_idx:
        con_sqlite_idx.close()

    # Save partial results after each SF round
    df_results = pd.DataFrame(results_data)
    df_results.to_csv("results.csv", index=False)
    log_print(f"\n  Results saved to results.csv")

    # Generate/update plots after each SF round
    generate_plots()

def run_queries(size_duck, size_sqlite_no, size_sqlite_idx, sf, sf_idx, q_num, results_data,
                con_duck, con_sqlite_no, con_sqlite_idx):
    log_print(f"\n--- Query #{q_num} with SF={sf} ---")

    try:
        query = get_tpch_query(q_num)
    except Exception as e:
        log_print(f"  Skipping Query #{q_num}: Could not get query text: {e}")
        return

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run on DuckDB
    _, df_duck = add_to_results(con_duck, query, results_data, q_num, "DuckDB", sf, size_duck, current_time_str)

    # Check if query should be skipped for SQLite (queries 7, 8, 9, 13)
    if q_num in SQLITE_SKIP_QUERIES:
        log_print(f"SQLite: SKIPPED (Query {q_num} not supported for SQLite)")
        return

    # Run on SQLite No Index
    df_sqlite_no = None
    if con_sqlite_no:
        _, df_sqlite_no = add_to_results(con_sqlite_no, query, results_data, q_num, "SQLite_No_Index",
                                          sf, size_sqlite_no, current_time_str)

    # Run on SQLite With Index
    if con_sqlite_idx:
        add_to_results(con_sqlite_idx, query, results_data, q_num, "SQLite_With_Index",
                       sf, size_sqlite_idx, current_time_str)

    # Compare results for first 4 SF rounds
    if sf_idx < COMPARE_RESULTS_ROUNDS and df_duck is not None and df_sqlite_no is not None:
        compare_results(df_duck, df_sqlite_no, q_num, "SQLite_No_Index")

def add_to_results(con, query, results_data, q_num, configuration_name, sf,
                   size_con, current_time_str):
    """
    Execute query and add results.
    Returns (exceeded_90s, result_df) tuple.
    """
    global skip_queries

    is_duckdb = configuration_name == "DuckDB"

    # Check if this query should be skipped for this configuration
    if q_num in skip_queries[configuration_name]:
        log_print(f"{configuration_name}: SKIPPED (Query {q_num} exceeded 90s in previous round)")
        return False, None

    if "SQLite" in configuration_name:
        query = translate_query_sql(query)

    time_con, result_df = measure_query_time(con, query, is_duckdb=is_duckdb)

    if time_con is not None:
        log_print(f"  {configuration_name}: {time_con:.4f}s")
        results_data.append({
            "Query_Num": q_num,
            "Configuration": configuration_name,
            "Scale_Factor": sf,
            "DB_Size_Bytes": size_con,
            "Time_Median": time_con,
            "Timestamp": current_time_str
        })

        # Print query results as DataFrame
        if result_df is not None:
            log_print(f"  {configuration_name} Results (Query {q_num}):")
            log_print(result_df)

        # If query took >90s, mark it to skip in future rounds for this config
        if time_con > 90:
            log_print(f"  {configuration_name} Query {q_num} exceeded 90s! Skipping in future rounds.")
            skip_queries[configuration_name].add(q_num)
            return True, result_df

        return False, result_df
    else:
        log_print(f"  {configuration_name}: FAILED for Query #{q_num}")

    return False, None


def normalize_value(val):
    """Normalize a value for comparison - handles dates, numbers, strings."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    # Remove time component from dates like "2024-01-15 00:00:00" -> "2024-01-15"
    if " 00:00:00" in s:
        s = s.replace(" 00:00:00", "")
    return s

def compare_results(df_duck, df_sqlite, q_num, config_name):
    """
    Compare results between DuckDB and SQLite.
    Prints differences if any.
    """
    log_print(f"\n  Comparing results: DuckDB vs {config_name} for Query {q_num}")

    if df_duck is None or df_sqlite is None:
        log_print("    Cannot compare: One or both results are None")
        return

    if df_duck.empty or df_sqlite.empty:
        log_print("    Cannot compare: One or both results are empty")
        return

    # Check column counts
    if len(df_duck.columns) != len(df_sqlite.columns):
        log_print(f"    Column count mismatch: DuckDB={len(df_duck.columns)}, {config_name}={len(df_sqlite.columns)}")
        return

    # Check column names (case-insensitive)
    duck_cols = [c.lower() for c in df_duck.columns]
    sqlite_cols = [c.lower() for c in df_sqlite.columns]
    if duck_cols != sqlite_cols:
        log_print(f"    Column names differ:")
        log_print(f"      DuckDB: {list(df_duck.columns)}")
        log_print(f"      {config_name}: {list(df_sqlite.columns)}")

    # Check row counts
    if len(df_duck) != len(df_sqlite):
        log_print(f"    Row count mismatch: DuckDB={len(df_duck)}, {config_name}={len(df_sqlite)}")

    # Compare values
    df_duck_cmp = df_duck.reset_index(drop=True)
    df_sqlite_cmp = df_sqlite.reset_index(drop=True)
    df_duck_cmp.columns = [c.lower() for c in df_duck_cmp.columns]
    df_sqlite_cmp.columns = [c.lower() for c in df_sqlite_cmp.columns]

    try:
        if df_duck_cmp.shape == df_sqlite_cmp.shape:
            diff_found = False
            for col in df_duck_cmp.columns:
                try:
                    # Try numeric comparison first
                    duck_numeric = pd.to_numeric(df_duck_cmp[col], errors='coerce')
                    sqlite_numeric = pd.to_numeric(df_sqlite_cmp[col], errors='coerce')

                    if not duck_numeric.isna().all() and not sqlite_numeric.isna().all():
                        # Both columns are numeric - compare with relative tolerance (0.1%)
                        # Use: |a - b| <= max(|a|, |b|) * 0.001
                        diff = duck_numeric.subtract(sqlite_numeric).abs()
                        max_vals = pd.concat([duck_numeric.abs(), sqlite_numeric.abs()], axis=1).max(axis=1)
                        tolerance = max_vals * 0.001  # 0.1% relative tolerance
                        tolerance = tolerance.clip(lower=0.01)  # minimum 0.01 absolute tolerance
                        if not (diff <= tolerance).all():
                            diff_found = True
                            log_print(f"    Differences found in column '{col}'")
                    else:
                        # String comparison with normalization
                        duck_norm = df_duck_cmp[col].apply(normalize_value)
                        sqlite_norm = df_sqlite_cmp[col].apply(normalize_value)
                        if not duck_norm.equals(sqlite_norm):
                            diff_found = True
                            log_print(f"    Differences found in column '{col}'")
                except Exception:
                    pass

            if not diff_found:
                log_print("    Results match!")
        else:
            log_print(f"    Shape mismatch: DuckDB={df_duck_cmp.shape}, {config_name}={df_sqlite_cmp.shape}")
    except Exception as e:
        log_print(f"    Comparison error: {e}")


def generate_plots():
    """
    Reads the csv, then creates the graphs
    22 graphs for every query (plots/query_01.png - plots/query_22.png)
    1 graph for summary (plots/query_summary.png)
    """
    log_print("\nGenerating Plots...")
    os.makedirs("plots", exist_ok=True)

    csv_file = "results.csv"
    if not os.path.exists(csv_file):
        log_print(f"  Error: {csv_file} not found. Cannot plot.")
        return

    colors = {
        "DuckDB": "green",
        "SQLite_No_Index": "red",
        "SQLite_With_Index": "blue"
    }

    df = pd.read_csv(csv_file)
    if df.empty:
        log_print("  No data to plot.")
        return

    creates_graphs_queries(df, colors)
    creates_graph_summary(df, colors)

def creates_graphs_queries(df, colors):
    """Create individual graphs for each query with zero-padded filenames"""
    queries = df['Query_Num'].unique()
    for q_num in queries:
        plt.figure(figsize=(10, 6))

        q_data = df[df['Query_Num'] == q_num]

        for config in q_data['Configuration'].unique():
            subset = q_data[q_data['Configuration'] == config]
            subset = subset.sort_values(by="Scale_Factor")

            plt.plot(subset['Scale_Factor'], subset['Time_Median'],
                     marker='o', linestyle='-',
                     label=config, color=colors.get(config, 'black'))

        plt.title(f"Query {q_num} Performance")
        plt.xlabel("Scale Factor (Input Size)")
        plt.ylabel("Time (Seconds)")
        plt.legend()
        plt.grid(True)

        # Zero-padded filename: plots/query_01.png, plots/query_02.png, etc.
        plt.savefig(f"plots/query_{q_num:02d}.png")
        plt.close()

    log_print(f"  Saved {len(queries)} query plots (plots/query_01.png - plots/query_{max(queries):02d}.png)")

def creates_graph_summary(df, colors):
    """Create summary graph with average times across all queries"""
    plt.figure(figsize=(10, 6))

    summary_df = df.groupby(['Scale_Factor', 'Configuration'])['Time_Median'].mean().reset_index()

    for config in summary_df['Configuration'].unique():
        subset = summary_df[summary_df['Configuration'] == config]
        subset = subset.sort_values(by="Scale_Factor")

        plt.plot(subset['Scale_Factor'], subset['Time_Median'],
                 marker='s', linestyle='--', linewidth=2,
                 label=config, color=colors.get(config, 'black'))

    plt.title("Average Execution Time (Summary)")
    plt.xlabel("Scale Factor (Input Size)")
    plt.ylabel("Average Time (Seconds)")
    plt.legend()
    plt.grid(True)

    plt.savefig("plots/query_summary.png")
    plt.close()

    log_print(f"  Saved summary plot (plots/query_summary.png)")

def translate_query_sql(query_text):
    """
    Translates DuckDB queries to SQLite queries
    """
    translated_text = query_text
    try:
        translated_text = sqlglot.transpile(query_text, read="duckdb", write="sqlite")[0]
    except Exception:
        # If automatic translation fails, we fall back to regex or original
        pass

    fixed_query = re.sub(
        r"extract\s*\(\s*year\s+from\s+([a-zA-Z0-9_.]+)\s*\)",
        r"strftime('%Y', \1)",
        translated_text,
        flags=re.IGNORECASE
    )
    return fixed_query

if __name__ == "__main__":
    run_benchmark()