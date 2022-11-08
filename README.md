# A Data-Driven Anomaly Detection on SRF Cavities at the European XFEL
** **Antonin Sulc, Annika Eichler, Tim Wilksen (DESY, Hamburg)**

The European XFEL is currently operating with hundreds of superconducting radio frequency cavities. To be
able to minimize the downtimes, prevention of failures on the SRF cavities is crucial. In this paper, we propose
an anomaly detection approach based on a neural network model to predict occurrences of breakdowns on the
SRF cavities based on a model trained on historical data. We used our existing anomaly detection infrastructure
to get a subset of the stored data labeled as faulty. We experimented with different training losses to maximally
profit from the available data and trained a recurrent neural network that can predict a failure from a series of
pulses. The proposed model is using a tailored architecture with recurrent neural units and takes into account
the sequential nature of the problem which can generalize and predict a variety of failures that we have been
experiencing in operation.


## Paper
- [paper](https://accelconf.web.cern.ch/ipac2022/papers/tupopt062.pdf) 
- [poster](https://github.com/sulcantonin/BPM_IPAC22/blob/main/tupopt062_poster.pdf)
- [torch model](https://github.com/sulcantonin/BPM_IPAC22/blob/main/models/anomaly_epoch_0127) 
- [torch training_data](https://github.com/sulcantonin/BPM_IPAC22/blob/main/model/training_data.pickle)
- [torch shuffle](https://github.com/sulcantonin/BPM_IPAC22/blob/main/model/permutation.npy)
