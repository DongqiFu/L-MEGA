## Package Required
The code is written by Python 3.7, and necessary packages are listed below.
- numpy 1.14.6
- scipy 1.2.1

## Dataset
Dataset folder contains preprocessed Alpha and OTC networks.
- For example, alpha_t0.txt contains the Alpha graph at t0, and alpha_d0.txt contains updated edges which tranform alpha_t0.txt into alpha_t1.txt
- The data format follows (node_u, node_v, timestamp), the timestamp is encoded with Unix Timestamp. To convert it into datetime format, please use
```
import datetime
print(datetime.datetime.fromtimestamp(1398744000.0))
>>> 2014-04-28 23:00:00
```

## How to Run
- We set the triangle as the target motif being preserved from graph cuts.
- First, please run prereq.py to obtain the initial clustering result at timestamp t0.
- Second, please run main.py to call L-MEGA to track the clustering result from timestamp t1 to t10.
