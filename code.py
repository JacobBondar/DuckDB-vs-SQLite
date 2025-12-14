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
# Fail Fast flags
run_sqlite_no = True
run_sqlite_idx = True

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
    print("="*40)
    print("💻 System Information:")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("="*40)
    return info

def generate_datasets(sf):
    """
    Create data in DuckDB and then copies it to SQLite
    """
    global run_sqlite_no
    global run_sqlite_idx
    print(f"\n--- Generating Data for SF = {sf} ---")

    clear_old_files()

    # Create DuckDB and the data
    con_duck = duckdb.connect("db_duck.duckdb")
    con_duck.execute("INSTALL tpch; LOAD tpch;")
    con_duck.execute(f"CALL dbgen(sf={sf});")
    print(f"DuckDB data generated with sf={sf}!")

    if not run_sqlite_no and not run_sqlite_idx:
        print("Skipping SQLite generation (both configurations disabled).")
        con_duck.close()
        return

    # Create SQLite and data transport
    # Using Pandas to transport the data from DuckDB to SQLite
    con_sqlite = sqlite3.connect("db_sqlite_no_index.db")

    for table in TPCH_TABLES:
        print(f"Transferring table: {table}...", end="\r")
        # Creating Data Frame from DuckDB
        df = con_duck.execute(f"SELECT * FROM {table}").df()
        # Transfer to SQLite
        df.to_sql(table, con_sqlite, index=False)

    print("Data transferred to SQLite (No Index).")
    con_sqlite.close()

    if run_sqlite_idx:
        # Best way to copy from SQLite to SQLite
        shutil.copy("db_sqlite_no_index.db", "db_sqlite_with_index.db")
        print("Created SQLite copy for indexing.")

    # If we don't need db_sqlite_no_index, then delete to preserve memory
    if not run_sqlite_no:
        os.remove("db_sqlite_no_index.db")
        print("Removed SQLite No-Index file (not needed).")

def clear_old_files():
    """
    Clear old files
    :return: None
    """
    if os.path.exists("db_duck.duckdb"):
        os.remove("db_duck.duckdb")
    if os.path.exists("db_sqlite_no_index.db"):
        os.remove("db_sqlite_no_index.db")
    if os.path.exists("db_sqlite_with_index.db"):
        os.remove("db_sqlite_with_index.db")

def create_sqlite_indexes(con):
    """
    Creates indexes in columns for keys and dates.
    """
    print("Creating Indexes for SQLite...")
    cursor = con.cursor()

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_orderdate ON orders(o_orderdate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lineitem_shipdate ON lineitem(l_shipdate)")

    con.commit()
    print("Indexes created successfully.")

def measure_query_time(con, query):
    """
    Executes the query, measures time, then returns median.
    If run takes more than 60 sec, then it stops and return the result.
    """
    times = []

    # Executing 3 times the loop
    for i in range(3):
        start_time = time.time()

        try:
            con.execute(query).fetchall()

        except Exception as e:
            print(f"Query Failed: {e}")
            return None

        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)

        # If execution takes more than 60 sec, we will not execute more
        if duration > 60: break

    return statistics.median(times)

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
    global run_sqlite_no
    global run_sqlite_idx

    get_system_info()

    scale_factors = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    results_data = []

    for sf in scale_factors: run_scales(sf, results_data)

    print("\nBenchmark Completed!")
    df_results = pd.DataFrame(results_data)
    df_results.to_csv("benchmark_results_final.csv", index=False)
    print("Results saved to benchmark_results_final.csv")
    generate_plots()

def run_scales(sf, results_data):
    global run_sqlite_no
    global run_sqlite_idx
    print(f"\n\nStarting scale for: {sf}")

    try:
        generate_datasets(sf)
    except Exception as e:
        print(f"Failed to generate data for SF = {sf}: {e}")
        return

    size_duck = os.path.getsize("db_duck.duckdb")
    size_sqlite_no = os.path.getsize("db_sqlite_no_index.db") if run_sqlite_no else 0
    size_sqlite_idx = 0

    con_duck = duckdb.connect("db_duck.duckdb")
    con_sqlite_no = None
    con_sqlite_idx = None

    if run_sqlite_no: con_sqlite_no = sqlite3.connect("db_sqlite_no_index.db")

    if run_sqlite_idx:
        con_sqlite_idx = sqlite3.connect("db_sqlite_with_index.db")
        create_sqlite_indexes(con_sqlite_idx)
        size_sqlite_idx = os.path.getsize("db_sqlite_with_index.db")

    for q_num in range(1, 23):
        run_queries(size_duck, size_sqlite_no, size_sqlite_idx, sf, q_num, results_data,
                        con_duck, con_sqlite_no, con_sqlite_idx)

    con_duck.close()
    if con_sqlite_no: con_sqlite_no.close()
    if con_sqlite_idx: con_sqlite_idx.close()

    # This is for temporal result, so if the computer crushes, I will still be able to see the results until that point.
    df_results = pd.DataFrame(results_data)
    df_results.to_csv("benchmark_results_partial.csv", index=False)

def run_queries(size_duck, size_sqlite_no, size_sqlite_idx, sf, q_num, results_data,
                con_duck, con_sqlite_no, con_sqlite_idx):
    global run_sqlite_no
    global run_sqlite_idx
    print(f"\n--- Question #{q_num} with SF={sf} ---")

    try:
        query = get_tpch_query(q_num)
    except Exception as e:
        print(f"Skipping Question #{q_num}: Could not return query text: {e}")
        return

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    add_to_results(con_duck, query, results_data, q_num, "DuckDB", sf, size_duck, current_time_str)

    if run_sqlite_no:
        run_sqlite_no = add_to_results(con_sqlite_no, query, results_data, q_num, "SQLite_No_Index",
                                       sf, size_sqlite_no, current_time_str, run_sqlite_no)

    if run_sqlite_idx:
        run_sqlite_idx = add_to_results(con_sqlite_idx, query, results_data, q_num, "SQLite_With_Index",
                                        sf, size_sqlite_idx, current_time_str, run_sqlite_idx)

def add_to_results(con, query, results_data, q_num, configuration_name, sf,
                   size_con, current_time_str, run_sqlite=None):
    if "SQLite" in configuration_name: query = translate_query_sql(query)
    time_con = measure_query_time(con, query)
    if time_con:
        print(f"{configuration_name}: {time_con:.4f}s")
        results_data.append({
            "Query_Num": q_num,
            "Configuration": configuration_name,
            "Scale_Factor": sf,
            "DB_Size_Bytes": size_con,
            "Time_Median": time_con,
            "Timestamp": current_time_str
        })

        if run_sqlite and time_con > 60:
            print(f"{configuration_name} is too slow! Disabling for future scales!")
            run_sqlite = False

    else: print(f"{configuration_name}: FAILED for Question #{q_num}")
    return run_sqlite

def generate_plots():
    """
    Reads the csv, then creates the graphs
    22 graphs for every query
    1 graphs for summery
    """
    print("\nGenerating Plots...")

    csv_file = "benchmark_results_final.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Cannot plot.")
        return

    # Creates directory for all the graphs
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colors = {
        "DuckDB": "green",
        "SQLite_No_Index": "red",
        "SQLite_With_Index": "blue"
    }

    df = pd.read_csv(csv_file)
    creates_graphs_queries(df, colors, output_dir)
    creates_graph_summary(df, colors, output_dir)

def creates_graphs_queries(df, colors, output_dir):
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

        plt.savefig(f"{output_dir}/query_{q_num}.png")
        plt.close()

    print(f"Successfully saved {len(queries)} query plots!")

def creates_graph_summary(df, colors, output_dir):
    plt.figure(figsize=(10, 6))

    summary_df = df.groupby(['Scale_Factor', 'Configuration'])['Time_Median'].mean().reset_index()

    for config in summary_df['Configuration'].unique():
        subset = summary_df[summary_df['Configuration'] == config]
        subset = subset.sort_values(by="Scale_Factor")

        plt.plot(subset['Scale_Factor'], subset['Time_Median'],
                 marker='s', linestyle='--', linewidth=2,
                 label=config, color=colors.get(config, 'black'))

    plt.title("Average Execution Time (Summary - Query 23)")
    plt.xlabel("Scale Factor (Input Size)")
    plt.ylabel("Average Time (Seconds)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{output_dir}/query_23_summary.png")
    plt.close()

    print(f"Successfully saved summary plot!")

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