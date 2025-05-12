import matplotlib

matplotlib.use('Agg')  # Используем бэкенд Agg для рендеринга графиков в файлы
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

# === Load and encode dataset ===
df = pd.read_csv("fashion_dataset_200.csv").dropna()  # Удаляем строки с пропусками
features = ['gender', 'age_range', 'height_range', 'weight_range', 'body_type', 'occasion']
target = 'style'

encoded_df = df.copy()
encoders = {}
for col in features + [target]:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    encoders[col] = le

X = encoded_df[features]
y = encoded_df[target]

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# === Main function ===
def run_all_models(form_data):
    try:
        input_vector = [[
            encoders['gender'].transform([form_data['gender']])[0],
            encoders['age_range'].transform([form_data['age_range']])[0],
            encoders['height_range'].transform([form_data['height_range']])[0],
            encoders['weight_range'].transform([form_data['weight_range']])[0],
            encoders['body_type'].transform([form_data['body_type']])[0],
            encoders['occasion'].transform([form_data['occasion']])[0],
        ]]
    except KeyError:
        return {"Error": "Invalid input data"}, None

    input_vector_np = np.array(input_vector)
    results = {}
    graphs = {}

    # === Task A: Supervised Algorithms ===
    models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    for name, model in models.items():
        try:
            # Обучение модели
            model.fit(X_train, y_train)
            pred = model.predict(input_vector_np)[0]

            # Преобразование предсказания
            if name == "Linear Regression":
                pred = int(round(pred))
                pred = np.clip(pred, 0, len(encoders['style'].classes_) - 1)
            pred_label = encoders['style'].inverse_transform([int(pred)])[0]

            # Оценка точности
            y_pred = model.predict(X_test)
            if name == "Linear Regression":
                y_pred = np.clip(np.round(y_pred).astype(int), 0, len(encoders['style'].classes_) - 1)
            accuracy = accuracy_score(y_test, y_pred) * 100

            results[name] = {
                "Prediction": pred_label,
                "Accuracy": f"{accuracy:.2f}%"  # Точность в процентах
            }
        except Exception as e:
            results[name] = {"Prediction": f"Error: {str(e)}", "Accuracy": "N/A"}

    # === Visualization: Bar Chart ===
    try:
        model_names = list(results.keys())[:8]
        accuracies = [float(results[model]["Accuracy"].strip('%')) if "Accuracy" in results[model] else 0 for model in
                      model_names]

        fig, ax = plt.subplots()
        ax.bar(model_names, accuracies, color='skyblue')
        ax.set_title("Model Accuracies")
        ax.set_ylabel("Accuracy (%)")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        # Сохранение графика как изображение
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graphs["Accuracy"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close(fig)
    except Exception as e:
        graphs["Accuracy"] = None

    return results, graphs


@app.route('/', methods=['GET', 'POST'])
def home():
    results, graphs = {}, {}  # Задаем пустые словари по умолчанию
    if request.method == 'POST':
        results, graphs = run_all_models(request.form)

    # Отладочный вывод
    print("Results:", results)  # Для проверки значения

    return render_template('all_algorithms.html', results=results, graphs=graphs)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
