# pls-shaving
"Gene shaving" - like algorithm (inspired by Hastie et al. 2000) using supervised PLS instead of PCA

# Algorithm

Iteratively we regress a y vector on an X matrix using PLS regression and shave off a small fraction (parameter *alpha*, a float between 0 and 1) of the worst contributors to main principal component until the top k contributors are left (k is controlled by the *size* parameter)

We return those top contributors in regard to coefficients in the first principal component of the PLS regression

This is particularly well suited for recursive programming as designed here
