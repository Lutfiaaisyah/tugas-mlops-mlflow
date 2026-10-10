import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Hubungkan langsung ke database lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-Quality-Classification")

# Aktifkan Autologging
mlflow.sklearn.autolog()

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="LogisticRegression_Auto"):
    # Aman dari error karena algoritma bukan berbasis Tree
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    
    print("[Auto Log] Run autologging berhasil!")
