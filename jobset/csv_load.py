import os
import csv
import ast
import numpy as np

BASE_DIR = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
GENE_START = 0
GENE_END = 120

ROW_NUM = 6     # r
COL_NUM = 17    # c

OUTPUT_CSV = "order_type_summary.csv"

def read_processing_times(csv_path):
    times = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            for cell in row[1:]:
                if not cell:
                    continue
                try:
                    machine, time_str = ast.literal_eval(cell)
                    if machine != '0':
                        times.append(float(time_str))
                except Exception:
                    continue

    return times


results = []

for gene_id in range(GENE_START, GENE_END + 1):
    filename = f"[{ROW_NUM}]r[{COL_NUM}]c,{gene_id}gene.csv"
    filepath = os.path.join(BASE_DIR, filename)

    if not os.path.exists(filepath):
        continue

    times = read_processing_times(filepath)

    if len(times) <= 1:
        continue

    variance = np.var(times, ddof=1)

    results.append({
        "filename": filename,
        "gene": gene_id,
        "variance": variance
    })


variances = np.array([r["variance"] for r in results])

low_q = np.percentile(variances, 33)
high_q = np.percentile(variances, 66)

for r in results:
    v = r["variance"]
    if v >= high_q:
        r["order_type"] = "大きい注文"
    elif v <= low_q:
        r["order_type"] = "小さい注文"
    else:
        r["order_type"] = "一般注文"

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["gene", "filename", "variance", "order_type"])

    for r in results:
        writer.writerow([
            r["gene"],
            r["filename"],
            round(r["variance"], 2),
            r["order_type"]
        ])


print(f"大订单阈值（>=）: {high_q:.2f}")
print(f"小订单阈值（<=）: {low_q:.2f}")
print(f"结果已保存到: {OUTPUT_CSV}")
