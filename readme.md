### Single Models CV (switched to 5 fold cv)
r2: 0.5975 - fpipe01, xgb(max_depth=2, num_rounds=1006, eta=0.005), no ID (LB 0.551)
r2: 0.6004 - same as above but with ID remaining (979 rounds) with ID (LB 0.556)
r2: 0.5796 - same as above but max_depth=4, num_rounds=1250, eta=0.0045 (LB 0.56773)
r2: 0.5415 - fpipe1 with LGBR (5fold cv), {'learning_rate': 0.007, 'max_depth': 3, 'n_estimators': 300}
r2: 0.5525 - fpipe1 with LGBR, {'learning_rate': 0.008, 'max_depth': 2, 'n_estimators': 400} (LB 0.54852, Val 0.5395)
---- Removed max(y) outlier ----
cv: 0.5853 - fpipe1 with LGBR, {'learning_rate': 0.004, 'max_depth': 3, 'n_estimators': 1000}
cv: 0.5513 - fpipe1 with LassoLarsCV(normalize=True)
cv: 0.5523 - fpipe1 with ElasticNetCV(normalize=True)
cv: 0.5852 - fpipe1 with LGBR, {'learning_rate': 0.0045, 'max_depth': 2, 'n_estimators': 1050, 'subsample': 0.9}
cv: 0.5602 - fpipe1 with RF (max_depth=3, n_estimators=500)
cv: 0.5299 - fpipe1 with outlier, LGBR, max_depth=2, n_estimators=1050, learning_rate=0.0045, subsample=0.9
cv: 0.5319 - same but added std_by_cat with LGBR
cv: 0.5323 - same but with size_by_cat with LGBR
cv: 0.5332 - added one hot encoding of cat labels with LGBR (best CV?, 0.55511)
cv: 0.2072 - same but with XGB parameters from the script
cv: 0.6123 - 0.5332 with np.log1p(y) and LGBR (best CV, LB 0.55387)


### Ideas:
- Mean of y across binary - encoding?
- Encode categorical

## List of pipelines with models
### Pipe 1
label encoded cat
int features with ID
tSVD, pca, ica, gaussian, sparse random (n_comp = 12)
Grid search:
- XGBRegressor:
	- {'rgr__learning_rate': 0.02, 'rgr__max_depth': 1, 'rgr__n_estimators': 500}
	- cv: 0.53326331831469276
- RandomForest
	- {'rgr__max_depth': 3, 'rgr__n_estimators': 500}
	- cv: 0.55301741982145924
- GradientBoostingRegressor
	- {'rgr__learning_rate': 0.01, 'rgr__max_depth': 3, 'rgr__n_estimators': 300}
	- cv: 0.54148965890716727
- LGBMRegressor
	- {'rgr__learning_rate': 0.003, 'rgr__max_depth': 3, 'rgr__n_estimators': 900}
	- сv: 0.54605548953798178
	- after removing outlier
	- {'learning_rate': 0.004, 'max_depth': 3, 'n_estimators': 1000}
	- cv: 0.58706277752654834
- BaggingRegressor
	- {'rgr__n_estimators': 700}
	- cv: 0.24768229495275224
- Meta: LGBRegressor
	- {'learning_rate': 0.005, 'max_depth': 2, 'n_estimators': 600}
	- cv: 0.53227470695258916
- Meta: XGBRegressor
	- {'learning_rate': 0.015, 'max_depth': 2, 'n_estimators': 300}
	- cv: 0.54409368771805788
