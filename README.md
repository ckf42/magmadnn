# MagmaDNN GCN

This repo is forked from the C++ neural network library MagmaDNN (commit hash 73b67f4f4abf1da75265b747ff4b910d3686d1d7). 

In this repo, a graph convolution layer (or to be more specific the one by [Kipf & Welling](https://arxiv.org/abs/1609.02907)) is implemented along with several classes that support this layer. The implementation uses cuda cuBlas and cuSparse libraries and hence GPU is required. 

As this is forked prior to MagmaDNN version 1.0, this repo may lack certain features that appear after version 1.0. 

This repo may be merged into the origin MagmaDNN repo after code clean-up. 

_author:_ Kam Fai Chan

_origin author:_ Daniel Nichols

_origin co-author:_ Sedrick Keh
