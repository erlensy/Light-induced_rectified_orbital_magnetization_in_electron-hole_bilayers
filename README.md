# Code for "Light-induced rectified orbital magnetization in electron-hole bilayers"

[Zenodo DOI: 10.5281/zenodo.22863044](https://doi.org/10.5281/zenodo.22863044)

## Requirements
- CMake >= 3.10
- C++20 compiler
- Armadillo
- Python 3 with NumPy and Matplotlib

## Build and run
```bash
mkdir build
cd build
cmake ..
cmake --build .
./ife
```

The program generates output files in the `data/` directory.

## Plotting
To plot the results:
```bash
python3 src/reader.py
```

## Struve function
The C++ implementation of the zeroth-order Struve function was adapted from `struve.cpp` in the RIFeatures project by C. P. Bridge:
https://github.com/CPBridge/RIFeatures/blob/ca0a2d56e3c6d840b523572728bdc1bcf8284c4b/src/struve.cpp

RIFeatures attributes the original Struve-function implementation to J.-P. Moreau.
The underlying `STVH0` routine is associated with: S. Zhang and J. Jin, *Computation of Special Functions*, Wiley, 1996.

## License
This software is distributed under the GNU General Public License version 3 (GPLv3). See the `LICENSE` file for details.
