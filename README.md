# Churn Prediction Starter Project

Files:
- sample_churn_data.csv : synthetic dataset
- train_churn.py : training & evaluation script
- requirements.txt : required python packages

Run:
```
pip install -r requirements.txt
python train_churn.py --data sample_churn_data.csv --out_dir outputs
```

Outputs will be saved to `outputs/` including `best_model.joblib` and `metrics.txt`.
