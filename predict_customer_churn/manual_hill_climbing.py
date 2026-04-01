import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X = train.drop("id",axis=1)
y = X.pop("Churn")

def hill_climb_blend(oof_preds, y, max_iter=200, step=0.05 , columns = None):
    
    n_models = oof_preds.shape[1]
    
    # başlangıç: eşit ağırlık
    weights = np.ones(n_models) / n_models
    
    best_score = roc_auc_score(y, np.dot(oof_preds, weights))
    iterCount = 0
    for _ in range(max_iter):
        print("="*20)
        iterCount += 1
        print(iterCount)
        improved = False
        
        for i in range(n_models):
            
            for direction in [-1, 1]:
                
                new_weights = weights.copy()
                new_weights[i] += direction * step
                
                # negatif weight olmasın
                if new_weights[i] < 0:
                    continue
                
                # normalize
                new_weights = np.maximum(new_weights, 0)
                new_weights /= new_weights.sum()
                
                score = roc_auc_score(y, np.dot(oof_preds, new_weights))
                
                if score > best_score:
                    weights = new_weights
                    best_score = score
                    improved = True
                    print(best_score)
                    ws = dict(zip( columns , weights.tolist() ))
                    ws_sorted = dict(sorted(ws.items(), key=lambda x: x[1],reverse=True))
                    
                    for key in ws_sorted:
                        print(key,ws_sorted[key]) 
                    print("-"*10)

        
        if not improved:
            step *= 0.5
        
        if step < 1e-6:
            break
    
    return weights, best_score


X_raw_oof = pd.read_csv("./stack/X_raw_oof.csv")
X_numeric_oof = pd.read_csv("./stack/X_numeric_oof.csv")
X_1_oof = pd.read_csv("./stack/X_1_oof.csv")
X_1_new_features_oof = pd.read_csv("./stack/X_1_new_features_oof.csv")
X_categorical_1_oof = pd.read_csv("./stack/X_categorical_1_oof.csv")
X_categorical_2_oof = pd.read_csv("./stack/X_categorical_2_oof.csv")
X_categorical_3_oof = pd.read_csv("./stack/X_categorical_3_oof.csv")
X_2_oof = pd.read_csv("./stack/X2_oof.csv")
X_3_oof = pd.read_csv("./stack/X3_oof.csv")
X_3_new_features_oof = pd.read_csv("./stack/X_3_new_features_oof.csv")
X_linear_raw_ridge_oof = pd.read_csv("./stack/X_linear_raw_ridge_oof.csv")
X_linear_raw_lasso_oof = pd.read_csv("./stack/X_linear_raw_lasso_oof.csv")
X_linear_raw_elastic_oof = pd.read_csv("./stack/X_linear_raw_elastic_oof.csv")
X_4_oof = pd.read_csv("./stack/X_4_oof.csv")
X_5_oof = pd.read_csv("./stack/X_5_oof.csv")
X_6_oof = pd.read_csv("./stack/X_6_oof.csv")
X_raw_xgb_oof = pd.read_csv("./stack/X_raw_xgb_oof.csv")
X_7_oof = pd.read_csv("./stack/X_7_oof.csv")
X_8_oof = pd.read_csv("./stack/X_8_oof.csv")
X_9_oof = pd.read_csv("./stack/X_9_oof.csv")
X_10_oof = pd.read_csv("./stack/X_10_oof.csv")
X_11_oof = pd.read_csv("./stack/X_11_oof.csv")
X_12_oof = pd.read_csv("./stack/X_12_oof.csv")




X_raw_oof_test = pd.read_csv("./stack/X_raw_oof_test.csv")
X_numeric_oof_test = pd.read_csv("./stack/X_numeric_oof_test.csv")
X_1_oof_test = pd.read_csv("./stack/X_1_oof_test.csv")
X_1_new_features_oof_test = pd.read_csv("./stack/X_1_new_features_oof_test.csv")
X_categorical_1_oof_test = pd.read_csv("./stack/X_categorical_1_oof_test.csv")
X_categorical_2_oof_test = pd.read_csv("./stack/X_categorical_2_oof_test.csv")
X_categorical_3_oof_test = pd.read_csv("./stack/X_categorical_3_oof_test.csv")
X_2_oof_test = pd.read_csv("./stack/X2_oof_test.csv")
X_3_oof_test = pd.read_csv("./stack/X3_oof_test.csv")
X_3_new_features_oof_test = pd.read_csv("./stack/X_3_new_features_oof_test.csv")
X_linear_raw_ridge_oof_test = pd.read_csv("./stack/X_linear_raw_ridge_oof_test.csv")
X_linear_raw_lasso_oof_test = pd.read_csv("./stack/X_linear_raw_lasso_oof_test.csv")
X_linear_raw_elastic_oof_test = pd.read_csv("./stack/X_linear_raw_elastic_oof_test.csv")
X_4_oof_test = pd.read_csv("./stack/X_4_oof_test.csv")
X_5_oof_test = pd.read_csv("./stack/X_5_oof_test.csv")
X_6_oof_test = pd.read_csv("./stack/X_6_oof_test.csv")
X_raw_xgb_oof_test = pd.read_csv("./stack/X_raw_xgb_oof_test.csv")
X_7_oof_test = pd.read_csv("./stack/X_7_oof_test.csv")
X_8_oof_test = pd.read_csv("./stack/X_8_oof_test.csv")
X_9_oof_test = pd.read_csv("./stack/X_9_oof_test.csv")
X_10_oof_test = pd.read_csv("./stack/X_10_oof_test.csv")
X_11_oof_test = pd.read_csv("./stack/X_11_oof_test.csv")
X_12_oof_test = pd.read_csv("./stack/X_12_oof_test.csv")



X_oof_total = pd.concat([X_raw_oof,X_numeric_oof,X_1_oof,X_1_new_features_oof,X_categorical_1_oof,X_categorical_2_oof,X_categorical_3_oof,X_2_oof,X_3_oof,X_3_new_features_oof,X_linear_raw_ridge_oof,X_linear_raw_lasso_oof,X_linear_raw_elastic_oof , X_4_oof , X_5_oof , X_6_oof , X_raw_xgb_oof , X_7_oof , X_8_oof , X_9_oof , X_10_oof , X_11_oof,X_12_oof],axis=1)
X_oof_total.columns = ["X_raw_oof","X_numeric_oof","X_1_oof","X_1_new_features_oof","X_categorical_1_oof","X_categorical_2_oof","X_categorical_3_oof","X_2_oof" , "X_3_oof","X_3_new_features_oof","X_linear_raw_ridge_oof","X_linear_raw_lasso_oof","X_linear_raw_elastic_oof" , "X_4_oof" , "X_5_oof" , "X_6_oof" , "X_raw_xgb_oof","X_7_oof" , "X_8_oof" , "X_9_oof" , "X_10_oof" , "X_11_oof","X_12_oof"]

X_oof_test_total = pd.concat([X_raw_oof_test,X_numeric_oof_test,X_1_oof_test,X_1_new_features_oof_test,X_categorical_1_oof_test,X_categorical_2_oof_test,X_categorical_3_oof_test,X_2_oof_test,X_3_oof_test,X_3_new_features_oof_test,X_linear_raw_ridge_oof_test,X_linear_raw_lasso_oof_test,X_linear_raw_elastic_oof_test , X_4_oof_test , X_5_oof_test , X_6_oof_test , X_raw_xgb_oof_test,X_7_oof_test , X_8_oof_test , X_9_oof_test , X_10_oof_test , X_11_oof_test,X_12_oof_test],axis=1)
X_oof_test_total.columns = ["X_raw_oof","X_numeric_oof","X_1_oof","X_1_new_features_oof","X_categorical_1_oof","X_categorical_2_oof","X_categorical_3_oof","X_2_oof" , "X_3_oof","X_3_new_features_oof","X_linear_raw_ridge_oof","X_linear_raw_lasso_oof","X_linear_raw_elastic_oof","X_4_oof","X_5_oof" , "X_6_oof" , "X_raw_xgb_oof","X_7_oof", "X_8_oof" , "X_9_oof", "X_10_oof" , "X_11_oof","X_12_oof"]#["X_raw_oof_test","X_numeric_oof_test","X_1_oof_test","X_1_new_features_oof_test","X_categorical_1_oof_test","X_categorical_2_oof_test","X_categorical_3_oof_test"]


weights, score = hill_climb_blend(X_oof_total.values, y , max_iter=200, step=0.1 , columns = X_oof_total.columns )

print("Best CV:", score)
print("Weights:", weights)

# test için
final_test_pred = np.dot(X_oof_test_total.values, weights)

result = pd.DataFrame({'id': test['id'], 'Churn': final_test_pred})
result.to_csv("manual_hill.csv",index=False)