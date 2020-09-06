import os 
import pandas as pd 
from sklearn import model_selection

if __name__ == "__main__":
    input_path = ""
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True) #rearranges data and removes original index of data
    y = df.target.values 
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, 'kfold'] = fold_  #CHECK IT
    df.to_csv(os.path.join(input_path, 'train_folds.csv'), index=False)

















