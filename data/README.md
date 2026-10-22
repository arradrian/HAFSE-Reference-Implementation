# Data

Datasets are not redistributed in this repository.

* Tennessee Eastman (source domain): http://web.mit.edu/braatzgroup/TE_process.zip
* Two-Tank benchmark (TD1): https://cs2ac.upc.edu/en/training-benchmarks/cyber-attacks-benchmark-simulator
* Gas Pipeline Network (TD2): available from the authors of the corresponding reference upon request.

Expected input for `src/data.build_datasets`: a list of runs (arrays of shape
n_observations × d) and one class label per run (or one label per observation).
Splitting is performed at run level before window extraction.
