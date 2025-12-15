# Exercise 6: DuckDB vs SQLite Performance Comparison

## Course Information
- **Course**: Big Data Analysis / Hananel Perl
- **Institution**: Jerusalem College of Technology
- **Semester**: Fall 2025
- **Due Date**: Thursday, December 18, 2025 at 23:00

## Objective
Compare query execution times between DuckDB and SQLite using the TPC-H benchmark.

## Configurations Tested
The program tests three configurations:

| Config | Database | Description |
|--------|----------|-------------|
| A | DuckDB | Tables created in DuckDB database |
| B | SQLite | Tables without any indexes |
| C | SQLite | Tables with indexes on relevant columns |

## Requirements

### Python Environment
- Use Anaconda version: `Anaconda3-2025.06-0`
- Download from: https://repo.anaconda.com/archive/

### Required Packages
- DuckDB version 1.4.2
- Sqlglot version 28.1.0
- psutil, platform, datetime (for system info)
- matplotlib (for graphs)
- pandas (for data handling and CSV export)

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. The program will:
   - Generate TPC-H data using DuckDB's tpch extension
   - Transfer data to SQLite databases
   - Run all 22 TPC-H queries on each configuration
   - Save results to CSV and generate graphs

## Program Flow

1. **For each scale factor (SF):**
   - Generate TPC-H data using DuckDB
   - Create SQLite databases (with and without indexes)
   - Run all 22 queries on each configuration
   - Each query runs 3 times, median time is recorded
   - Save results to CSV
   - Generate/update graphs (23 images)
   - Delete database files before next round

2. **Scale Factors:**
   - Start from SF = 0.001
   - Increase each round (approximately 10 rounds total)
   - Total runtime target: ~30 minutes

## Query Execution Rules

### Timeout Rules
- **>90 seconds**: Stop running that query for that configuration from that point forward
- **>10 seconds**: Run only once (no need for 3 runs and median)

### SQLite-Skipped Queries
Queries 7, 8, 9, and 13 are only run on DuckDB (skipped for SQLite due to conversion complexity).

### Slow Queries Reference (SQLite without index)
From testing, these queries become slow at higher SF: 4, 5, 17, 19, 20, 21, 22

## Output Files

### Generated During Execution
| File | Description |
|------|-------------|
| `results.csv` | All query timing results |
| `query_01.png` - `query_22.png` | Performance graph for each query |
| `query_summary.png` | Summary graph with average times |
| `output.txt` | Console output log |

### CSV Columns
- Query number
- Configuration (A/B/C)
- Database size on disk
- Median execution time
- Timestamp of last run
- Scale Factor (SF)

## Graphs
- **X-axis**: Input size (Scale Factor)
- **Y-axis**: Execution time
- **Lines**: One line per configuration (different colors)
- **Total**: 23 graphs (22 queries + 1 summary)

## Result Comparison
- Compare DuckDB vs SQLite results for the first 4 SF rounds
- Verify column names and counts match
- Print any differences in values found

## System Information
The program prints system specifications:
- Computer type
- CPU information
- Memory (RAM) amount
- Other relevant hardware info

## Submission Contents
The ZIP file should contain:
1. `requirements.txt` - Python dependencies
2. `main.py` - Main Python script
3. `output.txt` - Program output log
4. `results.csv` - All timing results
5. 23 graph images (query_01.png through query_22.png + query_summary.png)
6. `README.md` - This file

**Note**: Do NOT include generated database files (.duckdb, .sqlite) in submission.

## Important Notes
- Close other applications during testing for accurate measurements
- The Python script must generate all required data (no pre-generated data)
- Code must be original and explainable
- Use `time.time()` or similar for measuring execution times
- Execute queries directly through DuckDB/SQLite, not through pandas
- Print query results as pandas DataFrames using `print(df)`

## References
- TPC-H Benchmark: https://www.tpc.org/tpch/
- DuckDB TPC-H Extension: https://duckdb.org/docs/stable/core_extensions/tpch
- System Info Code: https://thepythoncode.com/article/get-hardware-system-information-python
- Big-O Notation Reference: https://www.bigocheatsheet.com/
