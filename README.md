# MRFingerprintingRecon.jl


| **Documentation**         | **Paper**                   | **Build Status**                      |
|:------------------------- |:--------------------------- |:------------------------------------- |
| [![][docs-img]][docs-url] | [![][paper-img]][paper-url] | [![][gh-actions-img]][gh-actions-url] |
|                           | [![][arXiv-img]][arXiv-url] | [![][codecov-img]][codecov-url]       |


MRFingerprintingRecon.jl is a Julia package that implements--so far--the *low rank inversion* reconstruction described in the paper [Low rank alternating direction method of multipliers reconstruction for MR fingerprinting](https://doi.org/10.1002/mrm.26639). This package is still work in progress and the interface will likely change over time. The ultimate goal of this package is to reproduce the [Matlab code](https://bitbucket.org/asslaender/nyu_mrf_recon) of the ADMM algorithm to Julia.


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://JakobAsslaender.github.io/MRFingerprintingRecon.jl/stable

[gh-actions-img]: https://github.com/JakobAsslaender/MRFingerprintingRecon.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/JakobAsslaender/MRFingerprintingRecon.jl/actions

[codecov-img]: https://codecov.io/gh/JakobAsslaender/MRFingerprintingRecon.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JakobAsslaender/MRFingerprintingRecon.jl

[arXiv-img]: https://img.shields.io/badge/arXiv-2107.11000-blue.svg
[arXiv-url]: https://arxiv.org/pdf/1608.06974.pdf

[paper-img]: https://img.shields.io/badge/doi-10.1002/mrm.29071-blue.svg
[paper-url]: https://doi.org/10.1002/mrm.26639
