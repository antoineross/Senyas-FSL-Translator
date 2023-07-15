##### This file is for `training` the model. The steps are as follows:

# 1. `Import Dependencies`
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Conv1D

# 2. `Setup Folders` for Collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('../../MP_Data_NF/') 
    
# Actions that we try to detect
actions = np.array(['ako',  'bakit', 'F', 'hi', 'hindi', 'ikaw',  'kamusta', 'L', 'maganda', 'magandang umaga', 'N', 'O', 'oo', 'P', 'salamat'])

# number of videos we want to use for training
no_sequences = 216 # edit this to change the number of videos used for training

# Videos are going to be 30 frames in length
sequence_length = 30

# 3. `Preprocess` Data and `Create Labels and Features`
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
print('np.array(sequences).shape: ', np.array(sequences).shape)
print('np.array(labels).shape: ',np.array(labels).shape)

y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print('y_test shape:', y_test.shape)

# 4. Build and `Train` LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Model design and implementation
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(30,258)))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# train model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

# 5. `Save/Load` Weights
modelName = 'fsl.h5' # edit this to change the model name
model.save(modelName)

# 6. `Evaluate` Model using Confusion Matrix and Accuracy
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

# 7. Save the Model to tfjs

def modelToTFJS(model, modelName):
    import tensorflowjs as tfjs
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    # numpy expects the following
    np.bool = np.bool_
    np.object = object

    model.load_weights(modelName)
    output_dir = '' # edit this to change the output directory
    tfjs.converters.save_keras_model(model, output_dir)
    
# modelToTFJS(model, modelName)
