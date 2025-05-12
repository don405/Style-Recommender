from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import random
import os
from deepface import DeepFace

# === Загрузка и подготовка данных ===
df = pd.read_csv("fashion_dataset_200.csv")
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
model = DecisionTreeClassifier()
model.fit(X, y)

# === Flask-приложение ===
app = Flask(__name__)

motivations = [
    "✨ Be confident! This style is made just for you.",
    "🔥 Turn heads everywhere you go.",
    "💎 Style isn’t what you wear — it’s how you wear it.",
    "🌟 Your presence deserves the best outfit.",
    "🕶️ Let your look speak louder than words.",
    "💥 Ready to conquer the world in style?",
    "😎 You’re not just wearing clothes — you’re making a statement.",
    "🌈 Express yourself boldly. This look is you.",
    "🪞 Mirror will love what it sees!"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    style = None
    motivation = None
    age_status = None
    user_input = {}

    if request.method == 'POST':
        # === Обработка фото (только возраст через DeepFace) ===
        image = request.files.get('photo')
        if image:
            image_path = os.path.join("static", "uploaded.jpg")
            image.save(image_path)
            try:
                result = DeepFace.analyze(image_path, actions=['age'], enforce_detection=False)
                age = result[0]['age']
                if age < 25:
                    user_input['age_range'] = '18-25'
                elif age < 35:
                    user_input['age_range'] = '25-35'
                elif age < 45:
                    user_input['age_range'] = '35-45'
                else:
                    user_input['age_range'] = '45-60'
                age_status = f"🧓 Detected age: {int(age)} → Age range: {user_input['age_range']}"
            except Exception as e:
                user_input['age_range'] = request.form['age_range']
                age_status = f"🔴 Error detecting age: {str(e)}"
        else:
            user_input['age_range'] = request.form['age_range']

        # Остальные параметры — ручной ввод
        user_input['gender'] = request.form['gender']
        user_input['height_range'] = request.form['height_range']
        user_input['weight_range'] = request.form['weight_range']
        user_input['body_type'] = request.form['body_type']
        user_input['occasion'] = request.form['occasion']

        input_vector = [[
            encoders['gender'].transform([user_input['gender']])[0],
            encoders['age_range'].transform([user_input['age_range']])[0],
            encoders['height_range'].transform([user_input['height_range']])[0],
            encoders['weight_range'].transform([user_input['weight_range']])[0],
            encoders['body_type'].transform([user_input['body_type']])[0],
            encoders['occasion'].transform([user_input['occasion']])[0],
        ]]

        prediction = model.predict(input_vector)[0]
        style = encoders['style'].inverse_transform([prediction])[0]
        motivation = random.choice(motivations)

    return render_template('index.html', style=style, motivation=motivation, user_input=user_input, age_status=age_status)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
