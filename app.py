import os
from flask import Flask, request, render_template, redirect, url_for, flash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from kan_fer import KANFER2013, KANRAFDB, KALFER2013, KALRAFDB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = 'supersecretkey'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Инициализация моделей
MODELS = {
    "KAN-FER2013": KANFER2013(),
    "KAN-RAF-DB": KANRAFDB(),
    "KAL-FER2013": KALFER2013(),
    "KAL-RAF-DB": KALRAFDB()
}

# Перевод названий эмоций
EMOTION_TRANSLATIONS = {
    "angry": "Злость",
    "disgust": "Отвращение",
    "fear": "Страх",
    "happy": "Радость",
    "sad": "Грусть",
    "surprise": "Удивление",
    "neutral": "Нейтральное"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверяем, загружен ли файл
        if 'file' not in request.files:
            flash('Файл не загружен', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Не выбран файл', 'error')
            return redirect(request.url)
        
        # Проверяем расширение файла
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            flash('Пожалуйста, загрузите изображение в формате JPG или PNG', 'error')
            return redirect(request.url)
        
        # Проверяем, какие модели использовать
        selected_models = request.form.getlist('models')
        if not selected_models:
            flash('Выберите хотя бы одну модель', 'error')
            return redirect(request.url)
        
        try:
            # Обрабатываем изображение
            img = Image.open(file)
            
            # Создаем график
            fig, axes = plt.subplots(1, len(selected_models) + 1, figsize=(15, 5))
            if len(selected_models) == 0:
                axes = [axes]

            if img.mode == 'L':
                axes[0].imshow(np.array(img), cmap='gray')
            else:
                axes[0].imshow(img)
            axes[0].set_title("Исходное изображение")
            axes[0].axis("off")
            
            # Обрабатываем каждой моделью
            results = {}
            for i, model_name in enumerate(selected_models, 1):
                model = MODELS[model_name]
                predictions = model.predict(img)
                sorted_pred = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                
                # Переводим названия эмоций
                translated_pred = [(EMOTION_TRANSLATIONS.get(e, e), p) for e, p in sorted_pred]
                
                # Создаем столбчатую диаграмму
                emotions = [e for e, _ in translated_pred]
                probs = [p for _, p in translated_pred]
                colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
                
                axes[i].barh(emotions, probs, color=colors)
                axes[i].set_xlim(0, 1)
                axes[i].set_title(model_name)
                axes[i].set_xlabel("Вероятность")
                
                # Сохраняем результаты
                results[model_name] = translated_pred
            
            # Конвертируем график в base64 для отображения в HTML
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            plot_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return render_template('result.html', plot_img=plot_img, results=results)
        except Exception as e:
            flash(f'Ошибка при обработке изображения: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('index.html', models=MODELS.keys())

if __name__ == '__main__':
    app.run(debug=True) 