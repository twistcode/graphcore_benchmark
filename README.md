# Graphcore Internal Benchmark

This repository contains a example code of Tensorflow on graphcore Intelligence Processing Unit (IPU) on their [Poplar SDK](https://www.graphcore.ai/products/poplar)

> The code presented here requires using Poplar SDK 3.1.0, and has been tested using Ubuntu 20.04 and Python 3.8.10

## Workspace setup

---

This repo was tested with the following requirement:

+ Ubuntu 20.04.5 LTS
+ Python 3.8.10
+ NVIDIA Driver 525.60.11
+ CUDA 11.2
+ cuDNN 8.1.1

Our testbench is the following:

+ Intel Xeon Gold 6152, 3.7 GHz (88 cores)
+ Mobo WS-C621E-SAGE Series
+ 64 GiB DDR4 (2666MHz)
+ Samsung SSD 960 PRO 1TB
+ NVIDIA GeForce Titan V, 12GB
+ NVIDIA Quadro GV100, 32GB
+ Graphcore IPU

---

### <img width="30" src="https://user-images.githubusercontent.com/81682248/177352641-89d12db1-45df-4403-8308-c6b9015a027d.png"></a> Computer Vision <a name="cv"></a>

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Simple MNIST CovNet | Image Classification | Training & Inference | [TensorFlow 2] |

The result is as follows:

| Model | GV 100 | IPU | Titan V | CPU (88 Cores) |
| ------- | ------- |------- | ------- | ------- |
| Model Test Lost | 0.06271637231111526 | 0.05814420431852341 | 0.06043968349695206 | 0.0602235347032547 |
| Model Accuracy| 0.9811000227928162 | 0.9814703464508057 | 0.9800000190734863 | 0.9817000031471252 |
| Wall Time| 2m 0.190s | 1m 26.398s | 2m 0.039s | 4m 15.376s |
| Card Temperature | 51 C | 29 C | 58 C | N/A |

| Model | Domain | Type |Links |
| ------- | ------- |------- | ------- |
| Simple CNN | Image Classification | Inference | [TensorFlow 2] |

The result is as follows:

| Model | GV 100 | IPU | Titan V | CPU (88 Cores) |
| ------- | ------- |------- | ------- | ------- |
| Wall time for Load model + Inference 1 image | 10.515s | 5.574s | 9.878s | 9.897s |
