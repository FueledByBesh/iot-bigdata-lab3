| input | controller | avg_queue | avg_wait | p95_wait | avg_delay_all | p95_delay_all | worst_node_wait | stops_per_trip | throughput_vph | blocked_green_s | unfinished | avg_wait_std | avg_queue_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | Fixed (Webster) | 2.91 | 11.56 | 26.00 | 11.56 | 26.00 | 11.56 | 0.79 | 3620.53 | 0.00 | 28.00 | 0.04 | 0.02 |
| synthetic | FNN-GA | 2.81 | 11.19 | 27.00 | 11.19 | 27.00 | 11.19 | 0.76 | 3620.32 | 0.00 | 24.33 | 0.07 | 0.03 |
| synthetic | FNN-GA retuned | 2.77 | 11.03 | 26.00 | 11.03 | 26.00 | 11.03 | 0.77 | 3620.56 | 0.00 | 25.33 | 0.04 | 0.02 |
| real (METR-LA) | Fixed (Webster) | 11.50 | 45.75 | 147.00 | 272.24 | 1762.00 | 45.75 | 0.83 | 3619.82 | 0.00 | 24.00 | 1.12 | 0.30 |
| real (METR-LA) | FNN-GA | 3.91 | 15.54 | 47.00 | 15.55 | 47.00 | 15.54 | 0.78 | 3619.76 | 0.00 | 20.33 | 1.05 | 0.27 |
| real (METR-LA) | FNN-GA retuned | 3.86 | 15.34 | 46.33 | 15.37 | 46.33 | 15.34 | 0.79 | 3619.92 | 0.00 | 19.33 | 0.87 | 0.23 |
