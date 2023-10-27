# MRFingerprintingRecon.jl


| **Documentation**         | **Paper**                   | **Build Status**                      |
|:------------------------- |:--------------------------- |:------------------------------------- |
| [![][docs-img]][docs-url] | [![][paper-img]][paper-url] | [![][gh-actions-img]][gh-actions-url] |
|                           | [![][arXiv-img]][arXiv-url] | [![][codecov-img]][codecov-url]       |


MRFingerprintingRecon.jl is a Julia package that implements sub-space reconstructions as described by [Jon Tamir et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26102) and by us in the paper [Low-rank alternating direction method of multipliers reconstruction for MR fingerprinting](https://doi.org/10.1002/mrm.26639) as *low rank inversion*.

Currently, the package only supports non-Cartesian trajectories, but a Cartesian implementation is under active development. The package is still work in progress and the interface will likely change over time. The ultimate goal of this package is to provide a Julia implemenation of the low-rank ADMM algorithm, similar to our [Matlab package](https://bitbucket.org/asslaender/nyu_mrf_recon).


[docs-img]: https://img.shields.io/badge/docs-latest%20release-blue.svg
[docs-url]: https://MagneticResonanceImaging.github.io/MRFingerprintingRecon.jl/stable

[gh-actions-img]: https://github.com/MagneticResonanceImaging/MRFingerprintingRecon.jl/workflows/CI/badge.svg
[gh-actions-url]: https://github.com/MagneticResonanceImaging/MRFingerprintingRecon.jl/actions

[codecov-img]: https://codecov.io/gh/MagneticResonanceImaging/MRFingerprintingRecon.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/MagneticResonanceImaging/MRFingerprintingRecon.jl

[arXiv-img]: https://img.shields.io/badge/arXiv-1608.06974-blue.svg
[arXiv-url]: https://arxiv.org/pdf/1608.06974.pdf

[paper-img]: https://img.shields.io/badge/doi-10.1002/mrm.26639-blue.svg
[paper-url]: https://doi.org/10.1002/mrm.26639
