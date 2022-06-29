# IVL-Reliability-and-Detection

# Reliability Tests
reliabilityInferenceConMat.py 
Gives confusion matrices for MV, IMV, PMP, WIMV, and WPMP with correction and without
using a worker confidence threshold of 60%.

In line 334, change False in the labels function to True to do worker correction. Keep it False to not.

reliabilityInferencePlot.py
Returns plots for accuracy, recall, precision, and fscore for each inference method over an average.
No user input is needed.
May take a little bit of time to run.

# Anomaly Detection Tests
AdversarialDetection.py
Returns a correctness-confidence(cor-con) graph of anomalous workers.
No user input is needed.

ADMultiRun.py
Gives efficieny of detection methods for anomalies via averaging over 40 runs.
No user input is needed.
May take a little bit of time to run.

CorrectnessDetection.py
Returns a cor-con graph of attackers.
No user input is needed.

TimeDetection.py
Returns a series of cor-con graphs as IVLs are introduced and anomalies are detected.
No user input is needed.

ExcludeAttackers.py
Returns graph of F1-Score for including and excluding attackers on inference making
with an increasing number of attackers in the worker pool.
No user input is needed.
