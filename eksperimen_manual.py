import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Hubungkan langsung ke database lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-Quality-Classification")

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

params = {"max_depth": 5, "random_state": 42, "criterion": "entropy"}

with mlflow.start_run(run_name="DecisionTree_Manual"):
    mlflow.set_tags({"stage": "eksperimen_awal", "metode": "manual"})
    mlflow.log_params(params)
    
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)
    
    # Mematikan skops (memakai cloudpickle) agar 100% aman dari error Windows
    mlflow.sklearn.log_model(
        model, 
        "model_decision_tree",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    
    print(f"[Manual Log] Run berhasil! Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
