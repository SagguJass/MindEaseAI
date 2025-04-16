from tensorflow.keras.models import load_model

# Load the model
model = load_model("model.h5", compile=False)

# Print a summary of the model
model.summary()
