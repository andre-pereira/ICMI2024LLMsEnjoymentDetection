import pandas as pd
import numpy as np
import os
from tensorflow import keras
from keras.utils import to_categorical

input_folder = "ModelsData"
input_folder_turn_taking_features = "turn_taking_annotations"
video_features_file = "video_features.csv"    

def get_turn_taking_features():
    array_list_turn_taking_features = []
    array_list_turn_taking_features_shifted_by_1 = []
    sorted_files = sorted(os.listdir(input_folder_turn_taking_features))
    for filename in sorted_files:
        if filename.endswith(".tsv"):
            file_path = os.path.join(input_folder_turn_taking_features, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines = lines[1:]
                # with two decimal places
                features = [line.strip().split('\t') for line in lines]
                array_list_turn_taking_features.extend(features)
                # Add null values for the first turn
                null_values = [None] * (len(features[0]) - 1)
                array_list_turn_taking_features_shifted_by_1.append(null_values)
                # Shift the features by 1 but do not add the first element
                for i in range(1, len(features)):
                    array_list_turn_taking_features_shifted_by_1.append(features[i-1][1:])
    return np.array(array_list_turn_taking_features), np.array(array_list_turn_taking_features_shifted_by_1)

def load_video_features():
    video_features = pd.read_csv(video_features_file)
    video_features.set_index(['Participant', 'Turn'], inplace=True)
    video_features_shifted_by_1 = video_features.shift(1, fill_value=0)
    return video_features, video_features_shifted_by_1


def load_open_smile_features():
    open_smile_features = pd.read_csv("Opensmile_features.csv")
    open_smile_features[['participant', 'turn']] = open_smile_features['filename'].str.split('_', expand=True)
    open_smile_features = open_smile_features.drop(['filename'], axis=1)
    open_smile_features['participant'] = open_smile_features['participant'].str.strip('P').astype(int)
    open_smile_features['turn'] = open_smile_features['turn'].str.strip('T').astype(int)
    # Setting the new columns as indices
    open_smile_features.set_index(['participant', 'turn'], inplace=True)
    open_smile_features = open_smile_features.sort_index()
    open_smile_features_shifted_by_1 = open_smile_features.shift(1, fill_value=0)
    return open_smile_features, open_smile_features_shifted_by_1

def load_audio_features():
    audio_features = pd.read_csv("Audio_Features.csv")
    audio_features[['participant', 'turn']] = audio_features['filename'].str.split('_', expand=True)
    audio_features = audio_features.drop(['filename'], axis=1)
    audio_features['participant'] = audio_features['participant'].str.strip('P').astype(int)
    audio_features['turn'] = audio_features['turn'].str.strip('T').astype(int)
    # Setting the new columns as indices
    audio_features.set_index(['participant', 'turn'], inplace=True)
    audio_features = audio_features.sort_index()
    audio_features_shifted_by_1 = audio_features.shift(1, fill_value=0)
    return audio_features, audio_features_shifted_by_1


class ICMILoadAnnotatorData:
    def __init__(self, fileAnnotatorRatings):
        # Load audio data
        self.audio_features, self.audio_features_shifted_by_1 = load_audio_features()
        self.open_smile_features, self.open_smile_features_shifted_by_1 = load_open_smile_features()
        self.video_features, self.video_features_shifted_by_1 = load_video_features()
        
        # Load data
        self.dataset = pd.read_csv(fileAnnotatorRatings, header=None)
        self.setup_dataset()
        
        # Prepare data subsets for each coder
        self.Annot1Set, self.Annot1Overal = self.prepare_coder_set('Annot1')
        self.Annot2Set, self.Annot2Overal = self.prepare_coder_set('Annot2')
        self.Annot3Set, self.Annot3Overal = self.prepare_coder_set('Annot3')

        self.Annot1Set_concat = flatten_and_filter(self.Annot1Set)
        self.Annot2Set_concat = flatten_and_filter(self.Annot2Set)
        self.Annot3Set_concat = flatten_and_filter(self.Annot3Set)

        self.Annot1Melted = melt_data(self.Annot1Set, 'Annot1')
        self.Annot2Melted = melt_data(self.Annot2Set, 'Annot2') 
        self.Annot3Melted = melt_data(self.Annot3Set, 'Annot3')

        self.models_flattened_data = {}
        self.models_flattened_data['Annot1'] = self.Annot1Set_concat
        self.models_flattened_data['Annot2'] = self.Annot2Set_concat
        self.models_flattened_data['Annot3'] = self.Annot3Set_concat

        self.models_flattened_data_shifted_by_1 = {}
        self.models_flattened_data_shifted_by_1['Annot1'] = get_shifted_flattened_data(self.Annot1Set)
        self.models_flattened_data_shifted_by_1['Annot2'] = get_shifted_flattened_data(self.Annot2Set)
        self.models_flattened_data_shifted_by_1['Annot3'] = get_shifted_flattened_data(self.Annot3Set)

        file_list = os.listdir(input_folder)
        array_list_LLM_names = []
        self.LLM_model_features_data_LSTM = None
        self.LLM_model_features_data = None
        self.LLM_model_features_data_shifted_by_1 = None
        for filename in file_list:
            if os.path.isdir(f"{input_folder}/{filename}"):
                continue
            name = filename[0:-4]
            LLMExchange, LLMOveral, LLMFlattened, LLMMelted = get_dataframes_from_file(f"{input_folder}/{filename}")
            self.models_flattened_data[name] = LLMFlattened
            shifted_data = get_shifted_flattened_data(LLMExchange)
            self.models_flattened_data_shifted_by_1[name] = shifted_data
            if isinstance(self.LLM_model_features_data, np.ndarray) == False:
                self.LLM_model_features_data_LSTM = [LLMExchange]
                self.LLM_model_features_data = np.array(LLMFlattened).reshape(len(LLMFlattened),1)
                self.LLM_model_features_data_shifted_by_1 = np.array(shifted_data).reshape(len(shifted_data),1)
            else:
                self.LLM_model_features_data_LSTM.append(LLMExchange)
                self.LLM_model_features_data = np.hstack((self.LLM_model_features_data, LLMFlattened[:, np.newaxis]))
                self.LLM_model_features_data_shifted_by_1 = np.hstack((self.LLM_model_features_data_shifted_by_1, shifted_data[:, np.newaxis]))
        
        
        self.array_list_turn_taking_features, self.array_list_turn_taking_features_shifted_by_1 = get_turn_taking_features()



    def get_Y(self):
        return np.concatenate((self.models_flattened_data['Annot1'], self.models_flattened_data['Annot2'], self.models_flattened_data['Annot3']))
    
    def get_Y_for_LSTM(self):
        y = self.Annot1Set.values
        y = np.concatenate((y, self.Annot2Set.values))
        y = np.concatenate((y, self.Annot3Set.values))
        y = np.nan_to_num(y, nan=-1)
        return y

    def get_X_for_LSTM(self, llm_features = True, turn_taking_features = True, audio_features = True, open_smile = True, video_features = True):
        models_features = [df.values for df in self.LLM_model_features_data_LSTM]
        models_features = np.stack(models_features, axis=2)
        models_features = np.nan_to_num(models_features, nan=-1)
        
        flattenedIndex = 0
        
        len_turn_taking_features = len(self.array_list_turn_taking_features[0])
        turn_taking_lstm = np.zeros((25, 29, len_turn_taking_features))
        len_audio_features = len(self.audio_features.loc[1,1].values)
        audio_features_lstm = np.zeros((25, 29, len_audio_features))
        len_open_smile_features = len(self.open_smile_features.loc[1,1].values)
        open_smile_lstm = np.zeros((25, 29, len_open_smile_features))
        len_video_features = len(self.video_features.loc[1,1].values)
        video_features_lstm = np.zeros((25, 29, len_video_features))
        p = 0
        participant_array = []
        for index, row in self.Annot1Set.iterrows():
            participant_array.append(index)
            index = int(index)
            # iterate all values in the row list in a for loop where 
            for t in range(len(self.Annot1Set.iloc[p])):
                # check if the value is not null
                if np.isnan(self.Annot1Set.iloc[p,t]):
                    turn_taking_lstm[p][t] = np.full(len_turn_taking_features, -1)
                    audio_features_lstm[p][t] = np.full(len_audio_features,-1)
                    open_smile_lstm[p][t] = np.full(len_open_smile_features,-1)
                    video_features_lstm[p][t] = np.full(len_video_features,-1)
                else:
                    turn_taking_lstm[p][t] = self.array_list_turn_taking_features[flattenedIndex]
                    audio_features_lstm[p][t] = self.audio_features.loc[index, t+1].values
                    open_smile_lstm[p][t] = self.open_smile_features.loc[index, t+1].values
                    video_features_lstm[p][t] = self.video_features.loc[index, t+1].values
                    flattenedIndex += 1
            p += 1
        tiled_x_arrays = []
        for index in range(1, 4):
            featuresToInclude = []
            featuresToInclude.append(custom_one_hot_encode(np.full((25,29,1), index), num_classes=3))
            if llm_features:
                featuresToInclude.append(models_features)
            if turn_taking_features:
                featuresToInclude.append(turn_taking_lstm)
            if audio_features:
                featuresToInclude.append(audio_features_lstm)
            if open_smile:
                featuresToInclude.append(open_smile_lstm)
            if video_features:
                featuresToInclude.append(video_features_lstm)
            tiled_x_arrays.append(np.concatenate(featuresToInclude, axis = 2))

        return np.concatenate(tiled_x_arrays, axis=0), np.tile(participant_array, 3)
            



    def get_X(self, llm_features = True, turn_taking_features = True, audio_features = True, open_smile = True, video_features = True, last_turn_features = True):
        X = []
        for i in range(1, 4):
            if i == 1:
                X_temp = np.full((len(self.models_flattened_data['Annot1']), 1), i)
            elif i == 2:
                X_temp = np.full((len(self.models_flattened_data['Annot2']), 1), i)
            elif i == 3:
                X_temp = np.full((len(self.models_flattened_data['Annot3']), 1), i)          

            if llm_features:
                X_temp = np.hstack((X_temp, self.LLM_model_features_data))
            if turn_taking_features:
                X_temp = np.hstack((X_temp, self.array_list_turn_taking_features))
            if audio_features:
                X_temp = np.hstack((X_temp, self.audio_features))
            if open_smile:
                X_temp = np.hstack((X_temp, self.open_smile_features))
            if video_features:
                X_temp = np.hstack((X_temp, self.video_features))
            
            if(last_turn_features):
                if llm_features:
                    X_temp = np.hstack((X_temp, self.LLM_model_features_data_shifted_by_1))
                if turn_taking_features:
                    X_temp = np.hstack((X_temp, self.array_list_turn_taking_features_shifted_by_1))
                if audio_features:
                    X_temp = np.hstack((X_temp, self.audio_features_shifted_by_1))
                if open_smile:
                    X_temp = np.hstack((X_temp, self.open_smile_features_shifted_by_1))
                if video_features:
                    X_temp = np.hstack((X_temp, self.video_features_shifted_by_1))

            if isinstance(X, np.ndarray) == False:
                X = X_temp
            else:
                X = np.vstack((X, X_temp))
        participant_array = np.array([])
        
        for index, row in self.Annot1Set.iterrows():
            non_null_values = row.notnull().sum()
            participant_array = np.append(participant_array,[index] * non_null_values)
        
        participant_array = np.tile(participant_array, 3)
        
        return X, participant_array

    def setup_dataset(self):
        self.dataset.columns = self.dataset.iloc[0]
        self.dataset = self.dataset.drop(self.dataset.index[0])
        self.dataset.columns.name = None
        
    def prepare_coder_set(self, coder_name):
        coder_set = self.dataset[self.dataset['Coder'] == coder_name]
        coder_set = coder_set.drop(['Coder'], axis=1)
        coder_overall = coder_set.iloc[:, [1]].copy()
        coder_overall = coder_overall.astype(float)
        coder_set.set_index('Participant', inplace=True)
        coder_set.drop(coder_set.columns[[0]], axis=1, inplace=True)
        coder_set = coder_set.astype(float)
        return coder_set, coder_overall
        
def flatten_and_filter(dataset):
    dataset_concat = dataset.values.flatten()
    return dataset_concat[pd.notna(dataset_concat)]
        
def melt_data(dataset, coder_name):
    melted = dataset.melt(var_name="Turn", value_name="Value")
    melted["Participant"] = melted.groupby("Turn").cumcount() + 1
    melted["Coder"] = coder_name
    return melted

def get_dataframes_from_file(file_path, delimiter='\t'):
    data = []
    max_columns = 0
    indices = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split(delimiter)
            if delimiter == ',':
                participant_number = float(row[0])
            else:
                participant_number = int(row[0].split('-P')[-1])
            indices.append(participant_number)
            row = row[1:]
            max_columns = max(max_columns, len(row))
            data.append(row)
    for row in data:
        row.extend([None] * (max_columns - len(row)))
    df = pd.DataFrame(data, columns=[f"Turn {i+1}" for i in range(max_columns)])   
    df.index = pd.Index(indices, name='Participant')
    if delimiter != ',':
        df = df.astype(float)
    else:
        df = df.apply(pd.to_numeric, errors='coerce')
    df_overal = pd.DataFrame(df.mean(axis=1).round().astype(int), columns=['Overal'])
    return df, df_overal, flatten_and_filter(df), melt_data(df, file_path.split('/')[-1][0:-4])

def get_shifted_flattened_data(data):
    shifted_data = data.apply(shift_row, axis=1)
    return flatten_and_filter(shifted_data)

# Function to shift row elements to the right without increasing row length
def shift_row(row):
    # Check if the row is already full (no NaNs)
    if row.notna().all():
        # If full, shift and drop the last value to maintain length
        shifted_values = [0] + row[:-1].tolist()
    else:
        # If not full, shift without adding new NaNs at the end
        non_na_values = row.dropna().tolist()
        non_na_values[-1] = np.nan
        shifted_values = [0] + non_na_values  # prepend NaN to shift right
        num_nans_to_add = len(row) - len(shifted_values)
        shifted_values += [np.nan] * num_nans_to_add
    
    return pd.Series(shifted_values, index=row.index)

def create_video_data():
    input_folder_video_features = "video_features"
    participant_array = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32]
    features = ['FaceRectX','FaceRectY','Pitch','Roll','Yaw','AU01','AU02','AU04','AU05','AU06','AU07','AU09','AU10','AU11','AU12','AU14','AU15','AU17','AU20','AU23','AU24','AU25','AU26','AU28','AU43','anger','disgust','fear','happiness','sadness','surprise','neutral']
    # Create a list for new column names for each statistic for each feature
    stat_types = ['first_quartile', 'median', 'third_quartile', 'iqr', 'mean_values', 'std_dev_values', 'min_values', 'max_values']
    new_columns = ['Participant', 'Turn'] + [f"{feature}_{stat}" for stat in stat_types for feature in features]

    videoData = pd.DataFrame(columns=new_columns)
    data_rows = []
    for i in participant_array:
        currentVideoFolder = f"{input_folder_video_features}/P{i}"
        sorted_files = sorted(os.listdir(currentVideoFolder))
        for filename in sorted_files:
            if filename.endswith(".csv"):
                df = pd.read_csv(f"{currentVideoFolder}/{filename}")
                df = df[features]
                name = filename.split('.')[0]
                turn = name.split('_')[1]
                turn = int(turn[1:])
                first_quartile = df.quantile(0.25)
                median = df.quantile(0.50)  # This is the same as df.median()
                third_quartile = df.quantile(0.75)
                iqr = third_quartile - first_quartile
                mean_values = df.mean()
                std_dev_values = df.std()
                min_values = df.min()
                max_values = df.max()
                # Flatten all statistics into a single list (order is important)
                stats = []
                for stat in [first_quartile, median, third_quartile, iqr, mean_values, std_dev_values, min_values, max_values]:
                    stats.extend(stat.tolist())

                # Create a DataFrame row with the correct number of elements
                row = pd.DataFrame([[i, turn] + stats], columns = new_columns)
                data_rows.append(row)

    videoData = pd.concat(data_rows, ignore_index=True)
    videoData.set_index(['Participant', 'Turn'], inplace=True)
    videoData = videoData.sort_index()
    videoData.to_csv("video_features.csv", index=True)

def custom_one_hot_encode(array, num_classes=3):
    result = np.zeros((array.shape[0], array.shape[1], num_classes))  # Adjusted shape
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            vector = array[i, j].flatten()  # Flatten each (1,) element
            result[i, j, vector - 1] = 1  # Index and assign
    return result