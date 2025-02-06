
import os
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import accuracy_score
from ICMIDataLoader import ICMILoadAnnotatorData, get_dataframes_from_file
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from sklearn.metrics import f1_score

annotatorData = ICMILoadAnnotatorData('Stats/Enjoyment dataset.csv')

# Data from the enjoyment self-reports from each user
user_satisfaction_ratings = [5,5,5,3,3,5,3,4,4,3,3,2,2,2,3,4,1,4,4,3,4,4,2,2,3] #inserted a 4 (average) instead of a null value
it_was_fun_talking_ratings = [4,5,5,4,4,5,3,4,5,4,4,2,2,2,5,5,4,3,4,5,4,5,4,4,4]
interesting_conversation_ratings = [4,4,5,2,4,5,3,4,3,4,4,3,1,1,4,2,2,4,3,3,4,4,2,3,2]
it_felt_strange = [3,5,3,1,2,5,2,4,4,2,3,3,1,2,4,3,1,4,3,3,3,5,2,5,2]

# Convert lists to NumPy arrays (handles None values)
dataUser = np.array([user_satisfaction_ratings, it_was_fun_talking_ratings, interesting_conversation_ratings, it_felt_strange], dtype=float)
average_ratings = np.nanmean(dataUser, axis=0)

Annot1_overal_ratings = [4,4,3,3,3,3,3,4,4,3,4,3,4,4,3,4,4,4,4,4,5,5,4,4,5]
Annot2_over_ratings = [3,3,2,3,3,4,3,4,4,3,4,3,4,4,3,4,4,4,5,4,5,4,3,4,4]
Annot3_overal_ratings = [3,4,2,3,3,4,3,3,4,3,4,4,4,3,3,5,4,4,4,4,5,4,3,4,4]

names = ['Annot1OveralAsked', 'Annot2OveralAsked', 'Annot3OveralAsked','AverageAsked', 'Annot1OveralAveraged', 'Annot2OveralAveraged', 'Annot3OveralAveraged', 'AverageAveraged']
AverageAsked = np.mean([annotatorData.Annot1Overal.values.flatten(), annotatorData.Annot2Overal.values.flatten(), annotatorData.Annot3Overal.values.flatten()], axis=0)
AverageAveraged = np.mean([Annot1_overal_ratings, Annot2_over_ratings, Annot3_overal_ratings], axis=0)
dataOveral = [annotatorData.Annot1Overal.values.flatten(), annotatorData.Annot2Overal.values.flatten(), annotatorData.Annot3Overal.values.flatten(),  AverageAsked, Annot1_overal_ratings, Annot2_over_ratings, Annot3_overal_ratings, AverageAveraged]
df_user_correlation = pd.DataFrame(columns=['LLM', 'PearsonRSatisfaction', 'PValueSatisfaction', 'PearsonRFun', 'PValueFun', 'PearsonRInteresting', 'PValueInteresting', 'PearsonRStrange', 'PValueStrange', 'PearsonRAll', 'PValueAll'])

for i in range(len(names)):
    pearsonSatisfaction, pValueSatisfaction = pearsonr(dataOveral[i], user_satisfaction_ratings)
    pearsonFun, pValueFun = pearsonr(dataOveral[i], it_was_fun_talking_ratings)
    pearsonInteresting, pValueInteresting = pearsonr(dataOveral[i], interesting_conversation_ratings)
    pearsonStrange, pValueStrange = pearsonr(dataOveral[i], it_felt_strange)
    pearsonAll, pValueAll = pearsonr(dataOveral[i], average_ratings)
    new_row = {'LLM': names[i], 'PearsonRSatisfaction': pearsonSatisfaction, 'PValueSatisfaction': pValueSatisfaction, 'PearsonRFun': pearsonFun, 'PValueFun': pValueFun, 'PearsonRInteresting': pearsonInteresting, 'PValueInteresting': pValueInteresting, 'PearsonRStrange': pearsonStrange, 'PValueStrange': pValueStrange, 'PearsonRAll': pearsonAll, 'PValueAll': pValueAll}
    df_user_correlation = pd.concat([df_user_correlation, pd.DataFrame([new_row])], ignore_index=True)

inputFolder = "ExchangesRatedByMachine"

file_list = os.listdir(inputFolder)

models_flattened_data = {}
models_flattened_data['Annot1'] = annotatorData.Annot1Set_concat
models_flattened_data['Annot2'] = annotatorData.Annot2Set_concat
models_flattened_data['Annot3'] = annotatorData.Annot3Set_concat
stacked_arrays_human = np.stack((annotatorData.Annot1Set_concat, annotatorData.Annot2Set_concat, annotatorData.Annot3Set_concat), axis=0) 

MeltedWithoutChatGPT = pd.concat([annotatorData.Annot1Melted, annotatorData.Annot2Melted, annotatorData.Annot3Melted])
MeltedOnlyLLMs = pd.DataFrame(columns=['Participant', 'Coder', 'Turn', 'Value'])

array_list_LLMs_flattened = []
array_list_LLMs_Melted = []
array_list_LLMs_Overal = []
for filename in file_list:
    if os.path.isdir(f"{inputFolder}/{filename}"):
        continue
    name = filename[0:-4]

    if filename[-3:] == 'tsv':
        LLMExchange, LLMOveral, LLMFlattened, LLMMelted = get_dataframes_from_file(f"{inputFolder}/{filename}", delimiter='\t')
    else:
        LLMExchange, LLMOveral, LLMFlattened, LLMMelted = get_dataframes_from_file(f"{inputFolder}/{filename}", delimiter=',')
    
    array_list_LLMs_flattened.append(LLMFlattened)
    array_list_LLMs_Melted.append(LLMMelted)
    array_list_LLMs_Overal.append(LLMOveral)
    models_flattened_data[name] = LLMFlattened
    
    if np.std(LLMOveral.values.flatten()) != 0:
        pearsonSatisfaction, pValueSatisfaction = pearsonr(LLMOveral.values.flatten(), user_satisfaction_ratings)
        pearsonFun, pValueFun = pearsonr(LLMOveral.values.flatten(), it_was_fun_talking_ratings)
        pearsonInteresting, pValueInteresting = pearsonr(LLMOveral.values.flatten(), interesting_conversation_ratings)
        pearsonStrange, pValueStrange = pearsonr(LLMOveral.values.flatten(), it_felt_strange)
        pearsonAll, pValueAll = pearsonr(LLMOveral.values.flatten(), average_ratings)
        new_row = {'LLM': name, 'PearsonRSatisfaction': pearsonSatisfaction, 'PValueSatisfaction': pValueSatisfaction, 'PearsonRFun': pearsonFun, 'PValueFun': pValueFun, 'PearsonRInteresting': pearsonInteresting, 'PValueInteresting': pValueInteresting, 'PearsonRStrange': pearsonStrange, 'PValueStrange': pValueStrange, 'PearsonRAll': pearsonAll, 'PValueAll': pValueAll}
        df_user_correlation = pd.concat([df_user_correlation, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_row = {'LLM': name, 'PearsonRSatisfaction': 1, 'PValueSatisfaction': 1, 'PearsonRFun': 1, 'PValueFun': 1, 'PearsonRInteresting': 1, 'PValueInteresting': 1, 'PearsonRStrange': 1, 'PValueStrange': 1, 'PearsonRAll': 1, 'PValueAll': 1}
        df_user_correlation = pd.concat([df_user_correlation, pd.DataFrame([new_row])], ignore_index=True)

pearsonSatisfaction, pValueSatisfaction = pearsonr(np.mean(array_list_LLMs_Overal, axis = 0).T[0], user_satisfaction_ratings)
pearsonFun, pValueFun = pearsonr(np.mean(array_list_LLMs_Overal, axis = 0).T[0], it_was_fun_talking_ratings)
pearsonInteresting, pValueInteresting = pearsonr(np.mean(array_list_LLMs_Overal, axis = 0).T[0], interesting_conversation_ratings)
pearsonStrange, pValueStrange = pearsonr(np.mean(array_list_LLMs_Overal, axis = 0).T[0], it_felt_strange)
pearsonAll, pValueAll = pearsonr(np.mean(array_list_LLMs_Overal, axis = 0).T[0], average_ratings)
new_row = {'AverageOfModelsOveral': names[i], 'PearsonRSatisfaction': pearsonSatisfaction, 'PValueSatisfaction': pValueSatisfaction, 'PearsonRFun': pearsonFun, 'PValueFun': pValueFun, 'PearsonRInteresting': pearsonInteresting, 'PValueInteresting': pValueInteresting, 'PearsonRStrange': pearsonStrange, 'PValueStrange': pValueStrange, 'PearsonRAll': pearsonAll, 'PValueAll': pValueAll}
df_user_correlation = pd.concat([df_user_correlation, pd.DataFrame([new_row])], ignore_index=True)

stacked_arrays_LLMs = np.stack(array_list_LLMs_flattened, axis=0)
models_flattened_data['MedianHuman'] = np.median(stacked_arrays_human, axis=0)
models_flattened_data['MedianLLM'] = np.median(stacked_arrays_LLMs, axis=0)

model_names = list(models_flattened_data.keys())
accuracy_df = pd.DataFrame(index=model_names, columns=model_names)
mae_df = pd.DataFrame(index=model_names, columns=model_names)
mse_df = pd.DataFrame(index=model_names, columns=model_names)
per_class_df = pd.DataFrame(index=model_names, columns=model_names)
df_weighted_average = pd.DataFrame(index = model_names, columns=model_names)
df_simple_average = pd.DataFrame(index = model_names, columns=model_names)

for key1, value1 in models_flattened_data.items():
    for key2, value2 in models_flattened_data.items():
        if key1 != key2:
            # calculate accuracy
            accuracy = accuracy_score(value1.astype(int), value2.astype(int))
            accuracy_df.at[key1, key2] = accuracy

            # calculate MAE
            mae = np.mean(np.abs(np.array(value1) - np.array(value2)))
            mae_df.at[key1, key2] = mae
            # calculate MSE
            mse = np.mean((np.array(value1) - np.array(value2))**2)
            mse_df.at[key1, key2] = mse
            
            cm = confusion_matrix(value1, value2, labels=[1,2,3,4,5])
            totals = np.sum(cm, axis=1)
            per_class_accuracies = np.divide(np.diag(cm), totals, out=np.zeros_like(totals, dtype=float), where=totals!=0)
            per_class_df.at[key1, key2] = per_class_accuracies
            
            # calculate weighted average f1 score from sci-kit learn
            weighted_average_f1 = f1_score(value1, value2, average='macro')
            df_weighted_average.at[key1, key2] = weighted_average_f1
            
            #calculate simple average
            simple_average = np.mean(per_class_accuracies)
            df_simple_average.at[key1, key2] = simple_average

accuracy_df.to_excel("Stats/ICMIStatsAccuracy.xlsx")
mae_df.to_excel("Stats/ICMIStatsMAE.xlsx")
mse_df.to_excel("Stats/ICMIStatsMSE.xlsx")
per_class_df.to_excel("Stats/ICMIStatsPerClassAccuracies.xlsx")
df_user_correlation.to_excel("Stats/ICMIStatsUserCorrelation.xlsx")
df_weighted_average.to_excel("Stats/ICMIStatsF1Macro.xlsx")
df_simple_average.to_excel("Stats/ICMIStatsSimpleAverage.xlsx")

#now use use cm to get per class accuracies
