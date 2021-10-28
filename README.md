# Modeling and Simulating In-Memory Memristive Deep Learning Systems: An Overview of Current Efforts

Supplementary GitHub Repository containing code to conduct direct performance comparisons of `MemTorch` [1], `DNN+NeuroSim_V2.1` [2], and the `IBM Analog Hardware Acceleration Kit` [3] from _Modeling and Simulating In-Memory Memristive Deep Learning Systems: An Overview of Current Efforts_, which is currently under consideration for publication in Array.

## Abstract

Deep Learning (DL) systems have demonstrated unparalleled performance in many challenging
engineering applications. As the complexity of these systems inevitably increase, they require increased processing capabilities and consume larger amounts of power, which are not readily available in resource-constrained processors, such as Internet of Things (IoT) edge devices. Memristive In-Memory Computing (IMC) systems for DL, entitled Memristive Deep Learning Systems (MDLSs), that perform the computation and storage of repetitive operations in the same physical location using emerging memory devices, can be used to augment the performance of traditional DL architectures; massively reducing their power consumption and latency. However, memristive devices, such as Resistive Random-Access Memory (RRAM) and Phase-Change Memory (PCM), are difficult and cost-prohibitive to fabricate in small quantities, and are prone to various device non-idealities that must be accounted for. Consequently, the popularity of simulation frameworks, used to simulate MDLS prior to circuit-level realization, is burgeoning. In this paper, we provide a survey of existing simulation frameworks and related tools used to model large-scale MDLS. Moreover,we perform direct performance comparisons of modernized open-source simulation frameworks, and provide insights into future modeling and simulation strategies and approaches. We hope that this treatise is beneficial to the large computers and electrical engineering community, and can help readers better understand available tools and techniques for MDLS development.

[1] C. Lammie, W. Xiang, B. Linares-Barranco, M. R. Azghadi, MemTorch: An Open-source Simulation Framework for Memristive Deep Learning Systems, arXiv:2004.10971 [cs] (2021).

[2] X. Peng, S. Huang, H. Jiang, A. Lu, S. Yu, DNN+NeuroSim V2.0: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators for On-chip Training, IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2020).

[3] M. J. Rasch, D. Moreda, T. Gokmen, M. L. Gallo, F. Carta, C. Goldberg, K. E. Maghraoui, A. Sebastian, V. Narayanan, A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays, arXiv:2104.02184 [cs] (2021).
