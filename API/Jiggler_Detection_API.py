# from flask import Flask, request, jsonify
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# from datetime import datetime
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# app = Flask(__name__)

# # Load the pre-trained model
# model_path = "D:\\SentientGeeks\\Mouse NEW\\API\\mouse_jiggler_classifier_vgg16.h5"
# model = load_model(model_path)

# # Constants
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# INTERVAL_MS = 10 * 1000  # Interval for graph creation (10 seconds in milliseconds)

# # Output Directory
# output_dir = "Jiggler_mouse_movements_Graph_API"
# os.makedirs(output_dir, exist_ok=True)

# # Function to convert timestamps to milliseconds
# def convert_to_milliseconds(timestamp):
#     try:
#         dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
#         total_milliseconds = (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
#         return total_milliseconds
#     except ValueError:
#         return None

# # Function to preprocess an image for prediction
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array

# @app.route('/predict', methods=['POST'])
# def predict_mouse_movements():
#     try:
#         # Retrieve the uploaded file
#         if 'file' not in request.files:
#             return jsonify({"error": "No file provided"}), 400
#         file = request.files['file']

#         # Save the CSV file temporarily
#         temp_file_path = os.path.join(output_dir, "temp_mouse_data.csv")
#         file.save(temp_file_path)

#         # Load the dataset
#         data = pd.read_csv(temp_file_path)

#         # Add 'Seconds' column
#         data['Seconds'] = data['Timestamp'].apply(convert_to_milliseconds)
#         data = data.dropna(subset=['Seconds'])  # Drop rows with invalid timestamps
#         data = data.sort_values(by='Seconds').reset_index(drop=True)

#         # Initialize variables for plotting
#         start_time = data['Seconds'].min()
#         end_time = data['Seconds'].max()
#         predictions = []

#         # Generate graphs and predict
#         while start_time < end_time:
#             # Filter data for the current interval
#             subset = data[(data['Seconds'] >= start_time) & (data['Seconds'] < start_time + INTERVAL_MS)]

#             if not subset.empty:
#                 # Plot the mouse movements
#                 graph_path = os.path.join(output_dir, f'mouse_movements_{int(start_time)}.png')
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(subset['X'], subset['Y'], marker='o', label='Mouse Movement')
#                 plt.title(f'Mouse Movements from {start_time / 1000:.1f} to {(start_time + INTERVAL_MS) / 1000:.1f} seconds')
#                 plt.xlabel('X Coordinate')
#                 plt.ylabel('Y Coordinate')
#                 plt.legend()
#                 plt.grid(True)
#                 plt.savefig(graph_path)
#                 plt.close()

#                 # Preprocess the graph and make a prediction
#                 img_array = preprocess_image(graph_path)
#                 prediction = model.predict(img_array)
#                 label = "Jiggler" if prediction[0] > 0.5 else "Human-like"
#                 predictions.append({"start_time": int(start_time), "prediction": label})

#             # Move to the next interval
#             start_time += INTERVAL_MS

#         return jsonify({"predictions": predictions})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the pre-trained model
model_path = "D:\\SentientGeeks\\Mouse NEW\\API\\mouse_jiggler_classifier_vgg16.h5"
model = load_model(model_path)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
INTERVAL_MS = 10 * 1000  # Interval for graph creation (10 seconds in milliseconds)

# Output Directory
output_dir = "Jiggler_mouse_movements_Graph_API"
os.makedirs(output_dir, exist_ok=True)

# Function to convert timestamps to milliseconds
def convert_to_milliseconds(timestamp):
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
        total_milliseconds = (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
        return total_milliseconds
    except ValueError:
        return None

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict_mouse_movements():
    try:
        # Retrieve the uploaded file
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']

        # Save the CSV file temporarily
        temp_file_path = os.path.join(output_dir, "temp_mouse_data.csv")
        file.save(temp_file_path)

        # Load the dataset
        data = pd.read_csv(temp_file_path)

        # Add 'Seconds' column
        data['Seconds'] = data['Timestamp'].apply(convert_to_milliseconds)
        data = data.dropna(subset=['Seconds'])  # Drop rows with invalid timestamps
        data = data.sort_values(by='Seconds').reset_index(drop=True)

        # Initialize variables for plotting
        start_time = data['Seconds'].min()
        end_time = data['Seconds'].max()
        predictions = []

        # Generate graphs and predict
        while start_time < end_time:
            # Filter data for the current interval
            subset = data[(data['Seconds'] >= start_time) & (data['Seconds'] < start_time + INTERVAL_MS)]

            if not subset.empty:
                # Plot the mouse movements
                graph_path = os.path.join(output_dir, f'mouse_movements_{int(start_time)}.png')
                plt.figure(figsize=(10, 6))
                plt.plot(subset['X'], subset['Y'], marker='o', label='Mouse Movement')
                plt.title(f'Mouse Movements from {start_time / 1000:.1f} to {(start_time + INTERVAL_MS) / 1000:.1f} seconds')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.legend()
                plt.grid(True)
                plt.savefig(graph_path)
                plt.close()

                # Preprocess the graph and make a prediction
                img_array = preprocess_image(graph_path)
                prediction = model.predict(img_array)
                label = "Jiggler" if prediction[0] > 0.5 else "Human-like"
                predictions.append({"start_time": int(start_time), "prediction": label})

                # Print the prediction for this interval
                print(f"Time: {start_time / 1000:.1f} to {(start_time + INTERVAL_MS) / 1000:.1f} seconds - Prediction: {label}")

            # Move to the next interval
            start_time += INTERVAL_MS

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ensure proper cleanup of resources after request processing
@app.teardown_appcontext
def shutdown_session(exception=None):
    import tensorflow.keras.backend as K
    K.clear_session()

if __name__ == '__main__':
    app.run(debug=True)
