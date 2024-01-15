# Anomaly-Detection

In this research, we aim to build a model for detecting anomalies in Industrial Control Systems. Industrial Control Systems (ICS) find extensive application across various industries, commonly incorporating diverse sensors and embedded systems within a dedicated network for operation. 

ICS have transitioned from being isolated systems to interconnected frameworks that leverage contemporary communication technologies and protocols. This shift aims to enhance efficiency, reduce operational costs, and further enhance an organization's support model.
These solutions, however, face the risk of cyber-physical attacks on their network. Although system security is a thoroughly researched and documented field, there is still a possibility for attacks to go undetected. 
Anomaly detection is the identification of rare, suspicious deviations from standard patterns in data. These anomalies can be referred to as outliers, noise, novelties, or exceptions. It plays a crucial role in identifying issues like hacking, fraud, equipment malfunctions, and errors.

There are three main classes of anomaly detection techniques: unsupervised, semi-supervised, and supervised. The choice of method depends on the availability of labeled data.
Supervised techniques require a dataset with labeled 'normal' and 'abnormal' instances, involving training a classifier, but handling the class imbalance can be challenging.
Semi-supervised methods use labeled data to create a model of normal behavior and assess anomalies based on the model's likelihood.
Unsupervised methods detect anomalies in unlabeled data, assuming the majority of instances are normal, and identifying the least congruent instances.
