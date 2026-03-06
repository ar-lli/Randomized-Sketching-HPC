# README file

Sketched GMRES algorithms tested:
1. Gaussian
2. Hadamard
3. Rademacher
4. SRFT
5. CountSketch

## Script file list
1. fwht.m -> Fast Walsh-Hadamard Transform function.
2. my_gmres.m -> Standard GMRES implementation.
3. my_gmres_sketch.m -> Sketched GMRES implementation.
4. test_2d.m -> First tests.
5. test_Aeta_2d.m -> Test the relationship between the iteration number and the residual of the GMRES and various sketched GMRES algorithms for the Poisson matrix's conductive parameter.
6. test_v2_2d.m -> Test the relationship between the iteration number and the residual of the GMRES and various sketched GMRES algorithms for different sketching dimensions.

## Images list
1. pde2d_A_eta.ofig -> Octave figure produced by test_Aeta_2d test.
2. pde2d_A_eta.png -> PNG figure produced by test_Aeta_2d test.
3. pde_v2_2d.ofig -> Octave figure produced by test_v2_2d test.
4. pde_v2_2d.png -> PNG figure produced by test_v2_2d test.
