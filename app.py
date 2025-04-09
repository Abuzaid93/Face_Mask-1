# # app.py
# from flask import Flask, render_template, request, redirect, url_for
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import os
# from datetime import datetime

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# # Load pre-trained model
# model = load_model('/Users/abuzaid/Desktop/ml gfg/deepfake/face/project/model/face_mask_detection_model.h5')  # Update with your model path

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0
#     return np.expand_dims(img, axis=0)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return redirect(request.url)
    
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
    
#     if file and allowed_file(file.filename):
#         # Save uploaded file
#         filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
#         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(save_path)
        
#         # Preprocess and predict
#         processed_img = preprocess_image(save_path)
#         prediction = model.predict(processed_img)
#         result = "WITH MASK" if prediction[0][0] > 0.5 else "WITHOUT MASK"
#         confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
        
#         return render_template('result.html', 
#                              image_path=save_path,
#                              result=result,
#                              confidence=round(confidence * 100, 2))
    
#     return redirect(request.url)

# if __name__ == '__main__':
#     app.run(debug=True)







# # app.py
# from flask import Flask, render_template, request, redirect, url_for, flash
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import os
# from datetime import datetime

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.secret_key = 'some_secret_key'  # Needed for flash messages

# # Load pre-trained model
# model = load_model('/Users/abuzaid/Desktop/ml gfg/deepfake/face/project/model/face_mask_detection_model.h5')

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0
#     return np.expand_dims(img, axis=0)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         # Redirect to home instead of back to /predict
#         return redirect(url_for('home'))
    
#     file = request.files['file']
#     if file.filename == '':
#         # Redirect to home instead of back to /predict
#         return redirect(url_for('home'))
    
#     if file and allowed_file(file.filename):
#         # Ensure the upload folder exists
#         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
#         # Save uploaded file
#         filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
#         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(save_path)
        
#         # Preprocess and predict
#         processed_img = preprocess_image(save_path)
#         prediction = model.predict(processed_img)
#         result = "WITH MASK" if prediction[0][0] > 0.5 else "WITHOUT MASK"
#         confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
        
#         return render_template('result.html', 
#                              image_path=save_path,
#                              result=result,
#                              confidence=round(confidence * 100, 2))
    
#     # Redirect to home instead of back to /predict
#     return redirect(url_for('home'))

# if __name__ == '__main__':
#     # Ensure the uploads folder exists on startup
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)





# # app.py
# from flask import Flask, render_template, request, redirect, url_for, flash
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import os
# from datetime import datetime

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.secret_key = 'some_secret_key'  # Needed for flash messages

# # Load pre-trained model
# model = load_model('/Users/abuzaid/Desktop/ml gfg/deepfake/face/project/model/face_mask_detection_model.h5')

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0
#     return np.expand_dims(img, axis=0)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     # Handle GET request - just redirect to home
#     if request.method == 'GET':
#         return redirect(url_for('home'))
    
#     # Handle POST request
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(url_for('home'))
    
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(url_for('home'))
    
#     if not (file and allowed_file(file.filename)):
#         flash('Invalid file type. Please upload JPG, JPEG or PNG files only.')
#         return redirect(url_for('home'))
    
#     # Ensure the upload folder exists
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
#     # Save uploaded file
#     filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
#     save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(save_path)
    
#     try:
#         # Preprocess and predict
#         processed_img = preprocess_image(save_path)
#         prediction = model.predict(processed_img)
#         result = "WITH MASK" if prediction[0][0] > 0.5 else "WITHOUT MASK"
#         confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
        
#         return render_template('result.html', 
#                             image_path=save_path,
#                             result=result,
#                             confidence=round(confidence * 100, 2))
#     except Exception as e:
#         # If there's an error, log it and redirect to home
#         print(f"Error processing image: {e}")
#         flash('Error processing image')
#         return redirect(url_for('home'))

# if __name__ == '__main__':
#     # Ensure the uploads folder exists on startup
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     app.run(debug=True)




# app.py
# from flask import Flask, render_template, request, jsonify
# import cv2
# import numpy as np
# import os
# import tensorflow as tf
# from PIL import Image
# import base64

# # Initialize Flask app
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# # Create upload folder if it doesn't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load the saved model (you'll need to save your model after training)
# # model = tf.keras.models.load_model('face_mask_model.h5')

# # For demonstration purposes, we'll recreate the model architecture from your code
# # In production, you should save and load your trained model
# model = tf.keras.models.load_model('/Users/abuzaid/Desktop/ml gfg/deepfake/face/project/model/face_mask_detection_model.h5')

# # In a real application, you would load the weights:
# # model.load_weights('face_mask_model_weights.h5')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     if file:
#         # Save the uploaded file
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)
        
#         # Process the image
#         try:
#             # Read image and preprocess
#             input_image = cv2.imread(filepath)
#             input_image_resized = cv2.resize(input_image, (128, 128))
#             input_image_scaled = input_image_resized / 255
#             input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])
            
#             # Make prediction
#             input_prediction = model.predict(input_image_reshaped)
#             input_pred_label = np.argmax(input_prediction)
            
#             # Determine result
#             if input_pred_label == 1:
#                 result = "The person in the image is wearing a mask"
#                 status = "with_mask"
#             else:
#                 result = "The person in the image is not wearing a mask"
#                 status = "without_mask"
            
#             # Convert image to base64 for display
#             with open(filepath, "rb") as img_file:
#                 img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
#             return jsonify({
#                 'result': result,
#                 'status': status,
#                 'probability': float(input_prediction[0][input_pred_label]),
#                 'image': img_base64
#             })
            
#         except Exception as e:
#             return jsonify({'error': str(e)})
    
#     return jsonify({'error': 'Something went wrong'})

# if __name__ == '__main__':
#     app.run(debug=True)








from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved model (you'll need to save your model after training)
# model = tf.keras.models.load_model('face_mask_model.h5')

# For demonstration purposes, we'll recreate the model architecture from your code
# In production, you should save and load your trained model
# def create_model():
#     num_of_classes = 2
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
#     model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(64, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(num_of_classes, activation='sigmoid'))
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['acc'])
#     return model

# model = create_model()


model = tf.keras.models.load_model('/Users/abuzaid/Desktop/ml gfg/deepfake/face/project/model/face_mask_detection_model.h5from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved model (you'll need to save your model after training)
model = load_model('/Users/abuzaid/Desktop/ml gfg/deepfake/face/project/model/face_mask_detection_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        try:
            # Read image and preprocess
            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # Make prediction
            input_prediction = model.predict(img_array)
            input_pred_label = np.argmax(input_prediction)
            
            # Determine result
            if input_pred_label == 1:
                result = "The person in the image is wearing a mask"
                status = "with_mask"
            else:
                result = "The person in the image is not wearing a mask"
                status = "without_mask"
            
            # Convert image to base64 for display
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'result': result,
                'status': status,
                'probability': float(input_prediction[0][input_pred_label]),
                'image': img_base64
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Something went wrong'})

if __name__ == '__main__':
    app.run(debug=True)')

# In a real application, you would load the weights:
# model.load_weights('face_mask_model_weights.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        try:
            # Read image and preprocess
            input_image = cv2.imread(filepath)
            input_image_resized = cv2.resize(input_image, (128, 128))
            input_image_scaled = input_image_resized / 255
            input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])
            
            # Make prediction
            input_prediction = model.predict(input_image_reshaped)
            input_pred_label = np.argmax(input_prediction)
            
            # Determine result
            if input_pred_label == 1:
                result = "The person in the image is wearing a mask"
                status = "with_mask"
            else:
                result = "The person in the image is not wearing a mask"
                status = "without_mask"
            
            # Convert image to base64 for display
            with open(filepath, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'result': result,
                'status': status,
                'probability': float(input_prediction[0][input_pred_label]),
                'image': img_base64
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Something went wrong'})

if __name__ == '__main__':
    app.run(debug=True)