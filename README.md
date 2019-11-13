# anseicca
ANSEICCA: Ambient Noise Source Estimation by Inversion of Cross-Correlation Amplitudes. This is a seismological waveform inversion code for ambient noise source directivity estimation. It implements the technique described by:

Datta, A., Hanasoge, S., & Goudswaard, J. (2019). Finite‐frequency inversion of cross‐correlation amplitudes for ambient noise source directivity estimation. Journal of Geophysical Research: Solid Earth, 124, 6653– 6665. https://doi.org/10.1029/2019JB017602

A. Package contents and overview

1. There are two versions of this Python 2.7 code - serial and parallel, each with their own wrappers and core modules. Other modules are common to the two versions. The parallel code uses MPI for Python (mpi4py) and should be significantly faster than the serial version when solving a large inverse problem (large number of receivers/stations).
2. The serial code (wrapper) is "anseicca_wrapper_serial.py" and it uses the code module "hans2013_serial.py".
3. The parallel code (wrapper) is "anseicca_wrapper_parallel.py" and it uses the code module "hans2013_parallel.py".
4. The common modules are "anseicca_utils1.py" and "anseicca_utils2.py".
5. The last script in the package is "view_result_anseicca.py" which is used to visualize (plot) the results produced by the code. Final as well as intermediate results (for the iterative inverse solution) may be accessed and visualized.
6. EXAMPLES directory - contains input file(s) required by the code, making this repository self-contained. No external input is required to get a demo of the working code.

B. How to run the code

Simple command line usage:

1. To run the serial code: "python anseicca_wrapper_serial.py"
2. To run the parallel code: "mpirun -np <n> python anseicca_wrapper_parallel.py"; <n> is the number of processors to use, should be equal to the number of receivers/stations in the problem. NOTE: if running on an HPC cluster, this command can be put into a script to be run with a job scheduler such as PBS.
3. To visualize the results: "python view_result_anseicca.py <file containing results produced by code>"

If you simply clone this repository and run the code on your system following the above instructions, it should run OK using input files from the EXAMPLES directory which have been hardwired into the two (serial and parallel) wrappers. To be able to use the code for your own purposes, you will need to read the description below.

C. Code description (user settings)
