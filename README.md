# PowerSGD
Practical Low-Rank Gradient Compression for Distributed Optimization

Abstract:
We study gradient compression methods to alleviate the communication bottleneck in data-parallel distributed optimization. Despite the significant attention received, current compression schemes either do not scale well or fail to achieve the target test accuracy. We propose a new low-rank gradient compressor based on power iteration that can i) compress gradients rapidly, ii) efficiently aggregate the compressed gradients using all-reduce, and iii) achieve test performance on par with SGD. The proposed algorithm is the only method evaluated that achieves consistent wall-clock speedups when benchmarked against regular SGD with an optimized communication backend. We demonstrate reduced training times for convolutional networks as well as LSTMs on common datasets.


# Reference
If you use this code, please cite the following [paper](https://arxiv.org/abs/1905.13727)

    @inproceedings{vkj2019powerSGD,
      author = {Thijs Vogels, Sai Praneeth Karimireddy, Martin Jaggi},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {arXiv},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
