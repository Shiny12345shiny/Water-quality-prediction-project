from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import json
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from pathlib import Path

import joblib
def algo(model):
    X_train=pd.read_csv(Path("balanced_processed_data\X_train.csv"))
    y_train=pd.read_csv(Path("balanced_processed_data\y_train.csv"))
    X_test=pd.read_csv(Path("balanced_processed_data\X_test.csv"))
    y_test=pd.read_csv(Path("balanced_processed_data\y_test.csv"))
    model=model
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print("____________________________________________________________________________________\n",model,"Report \n __________________________________________________________________________________")
    train_score=model.score(X_train,y_train)
    
    test_score=model.score(X_test,y_test)

    joblib.dump(model,Path("model\model_rfc.pkl"))
    path=Path("report\metrics.json")
    with open(path,'w') as f:
        scores={
           
            'train_score': train_score,
            'test_score': test_score
        }
    with open(path,'w') as f:
        json.dump(scores,f)

