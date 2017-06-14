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
- BaggingRegressor
	- {'rgr__n_estimators': 700}
	- cv: 0.24768229495275224
- Meta: LGBRegressor
	- {'learning_rate': 0.005, 'max_depth': 2, 'n_estimators': 600}
	- cv: 0.53227470695258916
- Meta: XGBRegressor
	- {'learning_rate': 0.015, 'max_depth': 2, 'n_estimators': 300}
	- cv: 0.54409368771805788
