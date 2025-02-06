import pandas as pd
import numpy as np
import xgboost as xgb
from ICMIDataLoader import ICMILoadAnnotatorData
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

annotator_data = ICMILoadAnnotatorData('Stats/Enjoyment dataset.csv')
outputFolder = "MultimodalMachineLearningModels/"
modelFileName = "XGBoost_"
y = annotator_data.get_Y()
y = y - 1

llm_features = ["llm_",True]
turn_taking_features = ["tt_",True]
audio_features = ["audio_",True]
open_smile = ["smile_",False]
video_features = ["video_",False]
last_turn_features = ["lasturn_",True]

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

X, participants = annotator_data.get_X(llm_features[1], turn_taking_features[1], audio_features[1], open_smile[1], video_features[1], last_turn_features[1])

X_Annot1 = X[0:590]
X_Annot2 = X[590:1180]
X_Annot3 = X[1180:1770]
y_Annot1 = y[0:590]
y_Annot2 = y[590:1180]
y_Annot3 = y[1180:1770]
data_rows = []

logo = LeaveOneGroupOut()
scaler = StandardScaler()
scaler.fit(X)

new_columns = ["coder", "participant"] + [f"turn_{i}" for i in range(1, 30)]
dfRatings = pd.DataFrame(columns=new_columns)
dfRatings = dfRatings.fillna(np.nan)

newTestSetDiffOrder = np.array([])
PredictionSetDiffOrder = np.empty((0, 5)) 

# Compute class weights
# enjoyment_classes = np.array([0., 1., 2., 3., 4.])
# class_weights = compute_class_weight('balanced', classes = enjoyment_classes, y=y)
# class_weight_dict = {cls: weight for cls, weight in zip(enjoyment_classes, class_weights)}

# def f1_eval(y_pred, dtrain):
#     y_true = dtrain.get_label()
#     preds = np.argmax(y_pred, axis=1)
#     f_score = f1_score(y_true, preds, average='macro')  # or 'micro', 'weighted'
#     return 'f1', f_score

for train_index, test_index in logo.split(X, groups = participants):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #replace all null X values with 0
    X_train[X_train == None] = 0
    X_test[X_test == None] = 0

    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #weights = np.array([class_weight_dict[cls] for cls in y_train])
 
    # Create DMatrix for XGBoost
    # D_train = xgb.DMatrix(X_train_scaled, label=y_train, nthread=-1, weight=weights)
    D_train = xgb.DMatrix(X_train_scaled, label=y_train, nthread=-1)
    D_test = xgb.DMatrix(X_test_scaled, label=y_test, nthread=-1)

    # Train the model and optimize for f1 score
    model = xgb.train({
    'tree_method': 'hist',  # 'hist' is recommended for GPU
    'device': 'cuda',  # Using GPU
    'objective': 'multi:softprob',
    'num_class': 5,
    'random_state': 23,  # Setting the seed here
    'max_depth': 5,  # Reduced depth
    'min_child_weight': 5,  # Increased child weight
    'gamma': 0.2,  # Increased gamma
    'subsample': 0.7,
    # 'colsample_bytree': 0.7,
    'eta': 0.1
    # }, D_train, custom_metric = f1_eval, num_boost_round=50, evals=[(D_train, 'train')], early_stopping_rounds=50, maximize=True)
    }, D_train, num_boost_round=100, evals=[(D_train, 'train'), (D_test, 'validation')], eval_metric='mlogloss', maximize=False)
    
    y_pred =  model.predict(D_test)#model.predict(X_test_scaled)
    PredictionSetDiffOrder = np.append(PredictionSetDiffOrder, y_pred, axis=0)
    newTestSetDiffOrder = np.append(newTestSetDiffOrder, y_test)

    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred.astype(float)
    y_pred = y_pred + 1
    participant = float(participants[test_index[0]])
    index_Annot1_end = int(y_pred.size/3)
    index_Annot2_end = int(2*y_pred.size/3)
    Annot1_pred = y_pred[0:index_Annot1_end]
    Annot2_pred = y_pred[index_Annot1_end:index_Annot2_end]
    Annot3_pred = y_pred[index_Annot2_end:y_pred.size]

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

y = y + 1
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


# Binarize the output labels for multi-class
y_true_bin = label_binarize(newTestSetDiffOrder, classes=[0, 1, 2, 3, 4])
n_classes = y_true_bin.shape[1]

# Assuming y_scores is a matrix of probabilities or scores for each class
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], PredictionSetDiffOrder[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for the multi-class problem
colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
plt.figure(figsize=(10, 8))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.savefig(f'{outputFolder}{modelFileName}multi_class_roc_{stringFeatures}.png', bbox_inches='tight', dpi=300)