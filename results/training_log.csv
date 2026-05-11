# Experiment Logs

Date: April 20, 2026

This file consolidates experimental outputs provided by the project author for submission.

## Experiment 1: Ollama Local, 10,000 Documents

### Per-Query Results

| Query | Baseline (ms) | S1 Only (ms) | S1+S2 (ms) | S1+S2+S3 (ms) | S1 Overlap | S1+S2 Overlap | S1+S2+S3 Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| When was the first barbie movie released? | 15282.45 | 9939.18 | 16745.76 | 16834.07 | 0.67 | 0.83 | 0.83 |
| Who developed the C programming language? | 8214.55 | 9998.78 | 16879.92 | 16599.44 | 0.67 | 0.83 | 0.83 |
| What is retrieval-augmented generation? | 10697.67 | 9923.72 | 17081.24 | 17015.38 | 0.67 | 0.83 | 0.83 |

### Averages and Speedups

| Metric | Value |
|---|---:|
| Average Baseline (ms) | 11398.22 |
| Average S1 Only (ms) | 9953.89 |
| Average S1+S2 (ms) | 16902.31 |
| Average S1+S2+S3 (ms) | 16816.30 |
| Average S1 Overlap | 0.67 |
| Average S1+S2 Overlap | 0.83 |
| Average S1+S2+S3 Overlap | 0.83 |
| Speedup Baseline / S1 Only | 1.15x |
| Speedup Baseline / S1+S2 | 0.67x |
| Speedup Baseline / S1+S2+S3 | 0.68x |

## Experiment 2: Groq, 20,000 Documents

### Per-Query Results

| Query | Baseline (ms) | S1 Only (ms) | S1+S2 (ms) | S1+S2+S3 (ms) | S1 Overlap | S1+S2 Overlap | S1+S2+S3 Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| When was the first barbie movie released? | 1821.78 | 1399.84 | 59047.74 | 58376.73 | 0.67 | 0.83 | 0.83 |
| Who developed the C programming language? | 2190.27 | 2077.92 | 58471.10 | 58018.61 | 0.67 | 0.83 | 0.83 |
| What is retrieval-augmented generation? | 2450.66 | 1917.80 | 58741.82 | 58053.97 | 0.67 | 0.83 | 0.83 |

### Averages and Speedups

| Metric | Value |
|---|---:|
| Average Baseline (ms) | 2154.24 |
| Average S1 Only (ms) | 1798.52 |
| Average S1+S2 (ms) | 58753.55 |
| Average S1+S2+S3 (ms) | 58149.77 |
| Average S1 Overlap | 0.67 |
| Average S1+S2 Overlap | 0.83 |
| Average S1+S2+S3 Overlap | 0.83 |
| Speedup Baseline / S1 Only | 1.20x |
| Speedup Baseline / S1+S2 | 0.04x |
| Speedup Baseline / S1+S2+S3 | 0.04x |

## Experiment 3: Groq, 10,000 Documents

### Per-Query Results

| Query | Baseline (ms) | S1 Only (ms) | S1+S2 (ms) | S1+S2+S3 (ms) | S1 Overlap | S1+S2 Overlap | S1+S2+S3 Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| When was the first barbie movie released? | 1392.25 | 1129.29 | 59345.08 | 58896.05 | 0.67 | 0.83 | 0.83 |
| Who developed the C programming language? | 1370.49 | 1326.50 | 59412.51 | 58541.53 | 0.67 | 0.83 | 0.83 |
| What is retrieval-augmented generation? | 1726.56 | 2233.65 | 58214.13 | 58872.26 | 0.67 | 0.83 | 0.83 |

### Averages and Speedups

| Metric | Value |
|---|---:|
| Average Baseline (ms) | 1496.43 |
| Average S1 Only (ms) | 1563.15 |
| Average S1+S2 (ms) | 58990.57 |
| Average S1+S2+S3 (ms) | 58769.95 |
| Average S1 Overlap | 0.67 |
| Average S1+S2 Overlap | 0.83 |
| Average S1+S2+S3 Overlap | 0.83 |
| Speedup Baseline / S1 Only | 0.96x |
| Speedup Baseline / S1+S2 | 0.03x |
| Speedup Baseline / S1+S2+S3 | 0.03x |

## Experiment 4: Chunking Sensitivity Study (1,000 Raw Documents)

Goal: Measure effect of chunk size and stride on chunk volume and chunking latency.

| Chunk Size | Stride | Total Chunks Generated | Average Chunks / Doc | Latency (ms) |
|---:|---:|---:|---:|---:|
| 32 | 16 | 48,895 | 48.90 | 3417.03 |
| 64 | 32 | 23,704 | 23.70 | 3384.05 |
| 128 | 64 | 11,094 | 11.09 | 3641.60 |
| 256 | 128 | 4,800 | 4.80 | 3286.69 |
| 512 | 256 | 1,695 | 1.70 | N/A (not provided) |

## Notes

- Values are transcribed from user-provided benchmark outputs.
- Two rows had merged numeric fields in the raw text; they were interpreted as:
  - 58053.97 and 0.67
  - 58872.26 and 0.67
- This file is a reconstructed log for submission where original run artifact files were not saved.
