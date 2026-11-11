| input | controller | avg_queue | avg_wait | p95_wait | avg_delay_all | p95_delay_all | worst_node_wait | stops_per_trip | throughput_vph | blocked_green_s | unfinished | avg_wait_std | avg_queue_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | Fixed (Webster) | 3.46 | 13.37 | 27.00 | 40.10 | 76.00 | 16.53 | 2.43 | 11159.71 | 0.00 | 262.00 | 0.01 | 0.00 |
| synthetic | FNN-GA naive | 2.57 | 9.96 | 26.00 | 29.84 | 61.00 | 10.26 | 2.29 | 11160.46 | 0.00 | 225.50 | 0.01 | 0.01 |
| synthetic | FNN-GA coordinated | 2.57 | 9.96 | 26.00 | 29.84 | 61.00 | 10.26 | 2.29 | 11160.46 | 0.00 | 225.50 | 0.01 | 0.01 |
| real (METR-LA) | Fixed (Webster) | 6.79 | 26.24 | 142.50 | 244.60 | 1406.50 | 36.32 | 2.51 | 11173.64 | 8.20 | 213.00 | 0.84 | 0.22 |
| real (METR-LA) | FNN-GA naive | 3.08 | 11.91 | 31.00 | 35.71 | 78.00 | 13.31 | 2.33 | 11174.35 | 0.01 | 178.00 | 0.07 | 0.02 |
| real (METR-LA) | FNN-GA coordinated | 3.10 | 11.97 | 31.00 | 35.88 | 79.50 | 13.57 | 2.33 | 11174.37 | 0.00 | 177.50 | 0.08 | 0.02 |
