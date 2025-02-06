import pandas as pd
import numpy as np
from ICMIDataLoader import ICMILoadAnnotatorData, get_dataframes_from_file, get_shifted_flattened_data
import seaborn as sns
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Dropout, TimeDistributed, Masking
from tf_keras.regularizers import l2
from tf_keras.initializers import Constant
from tf_keras.optimizers import Adam

from tf_keras.callbacks import EarlyStopping

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


os.environ["TF_USE_LEGACY_KERAS"]="1"
print(tf.config.list_physical_devices('GPU'))

annotator_data = ICMILoadAnnotatorData('Stats/Enjoyment dataset.csv')
outputFolder = "MultimodalMachineLearningModels/"
modelFileName = "LSTM_"
y = annotator_data.get_Y_for_LSTM()

llm_features = ["llm_",True]
turn_taking_features = ["tt_",True]
audio_features = ["audio_",True]
open_smile = ["smile_",True]
video_features = ["video_",True]
last_turn_features = ["lasturn_",False]

stringFeatures = ""
if llm_features[1]:
    stringFeatures += llm_features[0]
if turn_taking_features[1]:
    stringFeatures += turn_taking_features[0]
if audio_features[1]:
    stringFeatures += audio_features[0]
if open_smile[1]:
    stringFeatures += open_smile[0]
if video_features[1]:
    stringFeatures += video_features[0]
if last_turn_features[1]:
    stringFeatures += last_turn_features[0]


new_columns = ["coder", "participant"] + [f"turn_{i}" for i in range(1, 30)]
dfRatings = pd.DataFrame(columns=new_columns)
dfRatings = dfRatings.fillna(np.nan)

X, participants = annotator_data.get_X_for_LSTM(llm_features[1], turn_taking_features[1], audio_features[1], open_smile[1], video_features[1])
logo = LeaveOneGroupOut()

num_samples, num_timesteps, num_features = X.shape
X_reshaped = X.reshape(num_samples * num_timesteps, num_features)

scaler = StandardScaler()
#put -1 values as nan
X_reshaped = np.where(X_reshaped == -1, np.nan, X_reshaped)

# Fit on training data
X_reshaped = scaler.fit_transform(X_reshaped)

X_reshaped = np.nan_to_num(X_reshaped, nan=-1)

# Reshape back for LSTM (if you reshaped earlier)
X = X_reshaped.reshape(num_samples, num_timesteps, num_features)

data_rows = []

def masked_mse(y_true, y_pred):
    # Create a mask that selects only the non-padded values
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    # Use the mask to zero out the contribution of padded timesteps in the MSE computation
    squared_error = tf.square(y_true - y_pred) * mask
    # Compute the mean, but only consider non-zero entries in the mask
    return tf.reduce_sum(squared_error) / tf.reduce_sum(mask)


early_stopping = EarlyStopping(monitor='val_loss',  # Metric to monitor
                               min_delta=0.001,     # Minimum change to qualify as an improvement
                               patience=50,         # Number of epochs with no improvement after which training will be stopped
                               verbose=1,           # To print messages when stopping
                               mode='min',          # The direction is "minimize" for loss
                               restore_best_weights=True)  # Restore model weights from the epoch with the best value of the monitored quantity

for train_index, test_index in logo.split(X, groups = participants):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

    model = Sequential()    
    model.add(Masking(mask_value=-1, input_shape=(X.shape[1], X.shape[2])))  # Assuming 0 is your padding value
    model.add(LSTM(100, return_sequences=True, name='lstm2', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(LSTM(100, return_sequences=True, name='lstm3', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='linear',bias_initializer=Constant(value=3), kernel_regularizer=l2(0.001))))  # Output layer for regression ats each timestep
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss= masked_mse, metrics=[masked_mse])

    # Fit the model
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # predictp
    y_pred = model.predict(X_test)

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test) 
    print('Test Loss:', loss, 'Test MAE:', mae)
    
    y_pred = np.round(y_pred)
    participant = float(participants[test_index[0]])
    Annot1_pred = y_pred[0][y_test[0] != -1]
    Annot2_pred = y_pred[1][y_test[1] != -1]
    Annot3_pred = y_pred[2][y_test[2] != -1]

    print(Annot1_pred)
    print(Annot2_pred)
    print(Annot3_pred)
    row = [1.0, participant]
    row.extend(Annot1_pred)
    row.extend([None] * (len(new_columns) -2 - len(Annot1_pred)))
    row = pd.DataFrame([row], columns = new_columns)
    data_rows.append(row)
    row = [2.0, participant]
    row.extend(Annot2_pred)
    row.extend([None] *  (len(new_columns) -2 - len(Annot1_pred)))
    row = pd.DataFrame([row], columns = new_columns)
    data_rows.append(row)
    row = [3.0, participant]
    row.extend(Annot3_pred)
    row.extend([None] * (len(new_columns) -2 - len(Annot1_pred)))
    row = pd.DataFrame([row], columns = new_columns)
    data_rows.append(row)

dfRatings = pd.concat(data_rows, ignore_index=True)
dfRatings.set_index(['coder', 'participant'], inplace=True)
dfRatings = dfRatings.sort_index()
dfRatings.loc[1].to_csv(f"{outputFolder}{modelFileName}Annot1_ratings_{stringFeatures}.csv", header=False)
dfRatings.loc[2].to_csv(f"{outputFolder}{modelFileName}Annot2_ratings_{stringFeatures}.csv", header=False)
dfRatings.loc[3].to_csv(f"{outputFolder}{modelFileName}Annot3_ratings_{stringFeatures}.csv", header=False)


all_classes = [1,2,3,4,5]
y_pred = dfRatings.values.flatten()
y_pred = y_pred[pd.notna(y_pred)]

y = y.flatten()[y.flatten()!=-1]
# Compute confusion matrix
cm = confusion_matrix(y, y_pred, labels=all_classes)

# Plot using seaborn
plt.figure(figsize=(10, 7))  # Size of the figure
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')  # 'g' format to avoid scientific notation
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
# Save the figure
plt.savefig(f'{outputFolder}{modelFileName}confusion_matrix_{stringFeatures}.png', bbox_inches='tight', dpi=300)  # Adjust dpi for higher resolution

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
f1 = f1_score(y, y_pred, average='macro')
# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color='skyblue')
# Adding text labels above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Performance Metrics')
plt.ylim(0, 1)  # Assuming all metrics are between 0 and 1
plt.savefig(f'{outputFolder}{modelFileName}model_performance_metrics_{stringFeatures}.png', bbox_inches='tight', dpi=300)

class_totals = cm.sum(axis=1)
accuracies = np.zeros_like(class_totals, dtype=float)
for i in range(len(class_totals)):
    if class_totals[i] > 0:
        accuracies[i] = cm.diagonal()[i] / class_totals[i]  
    else:
        accuracies[i] = 0

per_class_accuracy = accuracies  # No need for np.nan_to_num since we control the output
print(f"Per-class accuracy: {per_class_accuracy}")
simple_average = np.mean(per_class_accuracy)

# print accuracy
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Simple average: {simple_average}")