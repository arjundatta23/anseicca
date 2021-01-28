# anseicca
ANSEICCA: Ambient Noise Source Estimation by Inversion of Cross-Correlation Amplitudes. This is a seismological waveform inversion (Python) code for ambient noise source directivity estimation. It implements the technique described by:

Datta, A., Hanasoge, S., & Goudswaard, J. (2019). Finite‐frequency inversion of cross‐correlation amplitudes for ambient noise source directivity estimation. Journal of Geophysical Research: Solid Earth, 124, 6653– 6665. https://doi.org/10.1029/2019JB017602

**********************************************************************************************
A. PACKAGE CONTENTS AND OVERVIEW

1. There are two versions of this code - serial and parallel - each with their own wrappers and core modules. Other modules are common to the two versions. The parallel code uses MPI for Python (mpi4py) and should be significantly faster than the serial version when solving a large inverse problem (large number of receivers/stations).
2. The serial code (wrapper) is "anseicca\_wrapper\_serial.py" and it uses the core module "hans2013\_serial.py".
3. The parallel code (wrapper) is "anseicca\_wrapper\_parallel.py" and it uses the core module "hans2013\_parallel.py".
4. The common modules are "anseicca\_utils1.py" and "anseicca\_utils2.py".
5. The last script in the package is "view\_result\_anseicca.py" which is used to visualize (plot) the results produced by the code. Final as well as intermediate results (for the iterative inverse solution) may be accessed and visualized.
6. EXAMPLES directory - contains input file(s) required by the code.

Finally, if using this code with a 1-D Earth model, you will need the 'SW1D_earthsr' set of modules, which is available as a separate repository: https://github.com/arjundatta23/SW1D_earthsr

**********************************************************************************************
B. HOW TO RUN THE CODE

Using a 0-D Earth model (2-D modelling domain):

1. To run the serial code: "python anseicca\_wrapper\_serial.py"
2. To run the parallel code: "mpiexec -np {n} python anseicca\_wrapper\_parallel.py"; {n} is the number of processors to use, should be equal to the number of receivers/stations in the problem. NOTE: if running on an HPC cluster, this command can be put into a script to be run with a job scheduler such as PBS.

Using a 1-D Earth model (3-D modelling domain):

On the command line, simply add three arguments: <mod\_file> <egn\_file> <disp\_file>. These are ASCII text files containing the 1-D Earth model, surface wave eigenfunctions and dispersion respectively (all in 'earthsr' format and compatible with the 'SW1D_earthsr' modules.)

3. To visualize the results: "python view\_result\_anseicca.py {arg1} {arg2 (OPTIONAL)}", where
	arg1 = pickle file/directory of files containing results produced by code;
	arg2 = coordinates file for stations or receivers

If you simply clone this repository and run the code on your system following the above instructions, it should run OK using input files from the EXAMPLES directory which have been hardwired into the two (serial and parallel) wrappers. To be able to use the code for your own purposes, you will need to read the description below.

**********************************************************************************************
C. CODE DESCRIPTION (USER SETTINGS)

The key tasks performed (sequentially) by the code, along with associated variables/parameters in the wrapper, are as follows:

1. Read in receiver/station location information (e.g. Easting/Northing, UTM coordinates) from a user-specified coordinates file which is pre-selected and hardwired into the code (variable "coordfile").
2. Place all the receivers on a uniform 2-D cartesian grid whose size and density are defined by the user ("hlbox\_outer" and "ngp\_inner").
3. Select a subset of receivers to work with, based on a relocation error threshold criterion ("glerr\_thresh"). The relocation is from actual coordinates to uniform grid coordinates.
4. Read the data (variable "infile") and prepare it for use by the code. This step is perfromed IF AND ONLY IF working with real data, as indicated by parameter "use_reald". The data is read and processed in the "anseicca\_utils2" module. I have worked with a specific format of input data used by Datta et al. (2019), however you can write your own class to suit your data and add it to "anseicca\_utils2".  
5. Run the central engine of the code, by calling the core "h13" module. This module is an extension of the work by Hanasoge (2013), and its usage is governed by the following key user settings:  
	(i) Whether working with synthetic or real data. If working with synthetic data, this is generated internally within the h13 module by forward modelling.
	(ii) The geometry of the problem (parameter "ratio_boxes"). In case of synthetic tests, the "true" sources may lie within or outside the inverse modelling domain, as shown by Figures 3 and B1 respectively, of Datta et al. (2019).  
	(iii) The geometry of model parameterization for the inverse problem (parameters "rad\_ring" and "w\_ring").  
	(iv) Whether to perform the inversion or not ("do_inv").  
	(v) Whether to force-stop the inversion after just one iteration or not ("iter_only1"). This option can be useful for testing, damping (L-curve) analysis etc.  

6. Store the results in Python binary archive (.npz) format, compatible with visualization script "view\_result\_anseicca.py".

Details of different parameters and how to set them are mentioned as comments in the serial code. To familiarize yourself with the code structure, I recommend doing synthetic tests ("use_reald"=False) and going through the above steps progressively:

1. Run the code without using the h13 module at all (comment out the line in the wrapper that calls "h13.inv_cc_amp"). You will see a map of the receiver network before and after gridding, along with an indication of the selected subset of receivers (parameter "map_plots" must be =True in order to see the plots).
2. Run the code with call to h13 included (undo action above) but with "do_inv" = False. You will see the setup of the inverse problem, i.e. the True and Starting source models.
3. Set "do\_inv"=True. First run with "iter\_only1"=True, then with "iter\_only1"=False. You will get the inversion result after only 1 iteration, and after natural convergence (unknown number of iterations), respectively.

**********************************************************************************************
D. VISUALIZATION OF RESULTS

The "view\_result\_anseicca.py" script has options for plotting:

1. The models (True/Starting/Inversion result)
2. The misfit kernels (computed in the starting model)
3. The inversion progress and summary (e.g. Figure 5 of Datta et al., 2019)

You can choose to plot any or all of the above by turning on or off the corresponding function calls at the bottom of the script.

**********************************************************************************************
REFERENCES

Datta, A., Hanasoge, S., & Goudswaard, J. (2019). Finite‐frequency inversion of cross‐correlation amplitudes for ambient noise source directivity estimation. Journal of Geophysical Research: Solid Earth, 124, 6653– 6665. https://doi.org/10.1029/2019JB017602.  
Hanasoge, S. M. (2013). The influence of noise sources on cross-correlation amplitudes. Geophysical Journal International, 192(1), 295–309. https://doi.org/10.1093/gji/ggs015.
