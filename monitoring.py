import optuna
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def objective(trial):
    # Hyperparameters to tune
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",  # XGBoost will report log loss after each iteration
        "eta": trial.suggest_float("eta", 1e-3, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Train model with a high num_boost_round and early stopping
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict probabilities on the validation set
    preds_proba = booster.predict(dvalid)

    # Compute the log loss on the validation data
    val_log_loss = log_loss(y_valid, preds_proba)

    # If we want to minimize log loss, we can either:
    #   1) Return val_log_loss and set direction="minimize" in the study
    #   2) Return -val_log_loss and set direction="maximize" (less common, but possible)
    return val_log_loss

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Optuna study to MINIMIZE log loss
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best trial:", study.best_trial.number)
print("Best hyperparameters:", study.best_params)
print("Best log loss:", study.best_value)

# Retrain final model with the best hyperparameters
best_params = study.best_params
best_params["objective"] = "binary:logistic"
best_params["eval_metric"] = "logloss"

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

final_booster = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dvalid, "validation")],
    early_stopping_rounds=50,
    verbose_eval=False
)

# Evaluate final model
preds_proba = final_booster.predict(dvalid)
final_log_loss = log_loss(y_valid, preds_proba)
print("Final model log loss:", final_log_loss)
