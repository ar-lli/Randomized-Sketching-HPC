# README file

Sketched GMRES algorithms tested:
1. Gaussian
2. Walsh-Hadamard
3. Discrete Cosine Transform
3. Rademacher
4. SRFT
5. CountSketch

## Script files list
All the test_ scripts compare the standard GMRES (no restarting) with the GMRES sketched versions outlined above.
### Function files
1. `fwht.m` -> Fast Walsh-Hadamard Transform function.
2. `my_gmres.m` -> Standard GMRES implementation.
3. `my_gmres_sketch.m` -> Sketched GMRES implementation.
### Test files
4. `test_k_2d.m` -> Iterations - Residuals plots as the sketching redunction parameter k varies. 
5. `test_Aeta_2d.m` ->  Iterations - Residuals plots as the Poisson matrix's conductive parameter eta varies.
6. `test_n_2d.m` -> Iterations - Residuals plots as the system matrix dimension n grows.
7. `test_n_2d_v2.m` -> Iterations - Residuals plots as the system matrix dimension n grows (with Hadamard and DCT transforms).
8. `test_n_dorr.m` -> As test_n_2d.m script but with 'dorr' matrix A (from gallery command).
9. `test_h_fixed_2d.m` -> Iterations - Residuals plots as Poisson system matrix grows and the h parameter stay fixed.
10. `test_sparse_suite.m` -> Iterations - Residuals plots as the matrix A downloaded from Matrix Sparse Suite varies.


## Images list
Pictures has the same basename of the script that produced it. png and ofig files are available.
1. test_k_2d.png
2. test_Aeta_2d.png
3. test_n_2d.png
4. test_n_2d_v2.png
5. test_n_dorr.png
6. test_h_fixed_2d.png
7. test_sparse_suite_*.png -> The asterisc is a placehorder for the sperimentation number of the test_sparse_suite script.

## SuiteSparse Matrix
The following are the sparse matrices downloaded from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/):
**2D/3D problem**
1. `poisson3Da.mat` (OK) -> n = 13,514; pattern symmetry: 100%; numeric symmetry: 0%; condition number: 1.121863e+03; min. singular value: 7.842040e-04.
2. `poisson3Db.mat` (TO TEST) -> n = 85,623; pattern symmetry: 100%; numeric symmetry: 0%.
3. `shermanACb.mat` (NO but it has a funny behaviour) -> n = 18,510; pattern symmetry: 14.9%; numeric symmetry: 2.7%; type: real; Condition number 1.795347e+07; min. singular value: 6.126961e-05.
4. `ck400.mat` (OK) -> n = 400; pattern symmetry: 98.9%; numeric symmetry: 0.2%; type: real; condition number: 2.592146e+05; min. singular value: 2.211310e-05.
**Subsequent Computational Fluid Dynamics Problem**
1. `cavity26.mat` (NO but same as shermanACb.mat) -> n = 4,562; pattern symmetry: 95.3%; numeric symmetry = 0%; type: real; condition number: 3.470374e+07; minimum sing. value: 

**other matrices** with poor results.
1. airfoil_2d.mat
2. e40r0100.mat
3. fs_680_1.mat
4. kim1.mat
5. mcca.mat
6. pde900.mat


_Note_. The _numeric symmetry_ attribute is defined as the percentage of non-zeros entries which are numerically symmetrical; the _pattern symmetry_ attribute is defined as the percentage of non-zeros entries which have a matching non-zero entry across the diagonal (but the value may be different).
