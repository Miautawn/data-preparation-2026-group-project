import pyarrow.parquet as pq
import pyarrow as pa
import os
import duckdb

def extract_top_rows_native(input_path, output_path, num_rows=10):
    parquet_file = pq.ParquetFile(input_path)
    first_batch = next(parquet_file.iter_batches(batch_size=num_rows))
    table = pa.Table.from_batches([first_batch])
    small_table = table.slice(0, length=num_rows)
    pq.write_table(small_table, output_path)
    print(f"Success! Created '{output_path}' with {len(small_table)} rows.")

input_file = r'file.parquet'
output_file = 'first_10_rows.parquet'

extract_top_rows_native(input_file, output_file, 10)

df = duckdb.query("SELECT * FROM 'first_10_rows.parquet' LIMIT 1").df()
print(df.T)