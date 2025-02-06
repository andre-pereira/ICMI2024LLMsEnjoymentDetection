import os
import time
from ICMIGeminiVertexAPI import ICMIGeminiVertexAPI
from ICMIOpenAIAPI import ICMIOpenAIAPI
from ICMIPromptsAndOptions import ICMIPromptsAndOptions


chosenAPI = ICMIGeminiVertexAPI()
#chosenAPI = ICMIOpenAIAPI()
modelName = chosenAPI.get_model_name()
sleepTime = 0

promptsAndOptions = ICMIPromptsAndOptions(includeSideVideos= False, 
                                          includeFrontalVideos= False, 
                                          includeScaleDetails = True,
                                          includeExamples = True,
                                          includeDialogHistory = True,
                                          includeResultHistory = True, 
                                          includeReflection= True)
baseFileName = promptsAndOptions.get_filename(modelName)
inputFolder = "annotations"


file_list = sorted(os.listdir(inputFolder), key=lambda x: int(x.split('.')[0]))
with open(f"{baseFileName}_PROMPTS.tsv", 'w', encoding='utf-8') as writeFilePrompts:
    with open(f"{baseFileName}_EXCHANGES.tsv", 'w', encoding='utf-8') as writeFileExchanges:
        with open(f"{baseFileName}_REASONING.tsv", 'w', encoding='utf-8') as writeFileReasoning:
            for filename in file_list:
                if filename.endswith(".txt"):
                    fileNumber = filename[:-4]
                    print(f"File number: {fileNumber}")
                    with open(os.path.join(inputFolder, filename), 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    turn = 1
                    historyWithRatings, historyWithoutRatings = "", ""
                    writeFileReasoning.write(f"{modelName}-P{filename[:-4]}\n")
                    writeFileExchanges.write(f"{modelName}-P{filename[:-4]}\t")
                    for line in lines:
                        split_line = line.split("\t")
                        ExchangeNumber = split_line[0]
                        ExchangeText = split_line[3]

                        prompt = promptsAndOptions.get_prompt(turn, ExchangeText, historyWithRatings, historyWithoutRatings)
                        
                        time.sleep(sleepTime)
                        rating = chosenAPI.rate_conversation(prompt, fileNumber, turn, promptsAndOptions.includeSideVideos, promptsAndOptions.includeFrontalVideos)
                        
                        ratingScoreOnly = rating[-1]
                        writeFilePrompts.write(f"{prompt} \n\n")
                        writeFileReasoning.write(f"{turn}\t{rating}\n")
                        writeFileExchanges.write(f"{ratingScoreOnly}\t")
                        historyWithRatings += f"[Exchange {ExchangeNumber}]: {ExchangeText}\t[Score] {ratingScoreOnly}"
                        historyWithoutRatings += f"[Exchange {ExchangeNumber}]: {ExchangeText}"
                        print(f"Turn: {turn} Rating: {rating}")
                        turn += 1
                    writeFileExchanges.write("\n")

