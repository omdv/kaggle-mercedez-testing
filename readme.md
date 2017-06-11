### Single Models CV
r2: 0.5975 - fpipe01, xgb(max_depth=2, num_rounds=1006, eta=0.005), no ID (LB 0.551)
r2: 0.6004 - same as above but with ID remaining (979 rounds) with ID (LB 0.556)
r2: 0.5796 - same as above but max_depth=4, num_rounds=1250, eta=0.0045 (LB 0.56773)


### Ideas:
- Mean of y across binary - encoding?
- Encode categorical

### Issues
- Cross-val is not representative or public LB is not representative?

## List of pipelines with models
### Pipe 1
label encoded cat
int features with ID
tSVD, pca, ica, gaussian, sparse random (n_comp = 12)
