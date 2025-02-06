import datetime

class ICMIPromptsAndOptions:
    def __init__(self, includeSideVideos, includeFrontalVideos, includeScaleDetails, includeExamples, includeDialogHistory, includeResultHistory, includeReflection):
        self.promptBeginning = "Given the following scale and the current exchange between a robot and a human, rate the user enjoyment in the current exchange with an integer value (1 to 5)."
        self.promptScale = """
        User enjoyment scale:
        1. Very low enjoyment - Discomfort and/or frustration
        2. Low enjoyment - Boredom or interaction failure
        3. Neutral enjoyment - Politely keeping up the interaction
        4. High enjoyment - Smooth and effortless interaction
        5. Very high enjoyment - Immersion in the conversation and/or deeper connection with the robot
        """
        self.promptScaleDetailsMultimodal = """Scale details:
        To rate the exchange higher on the user enjoyment scale (4 and 5), look for signs of enjoyment, such as smirking, movement, flow of conversation (the topic is moving forward), no strain or discomfort, asking questions [to the robot], smooth turn taking, dynamic tonality (and dynamic phrasing of sentences), being playful, sharing personal experiences [to the robot], sharing an understanding (common ground) [with the robot], and anthropomorphizing [the robot].
        To rate the exchange lower on the scale (1 and 2), look for signs of dis-enjoyment, such as low energy, sighing, tiredness, repeated questions [from the robot], long breaths, restless movements (i.e, adaptors, such as moving in the chair from side to side or changing arm position), flat tonality, silence, awkward and negative facial expressions, flaring nostrils, disengagement cues (e.g., turning away from the robot), and topic closure (e.g., “Let's talk about something else”).
        Neutral enjoyment (3) refers to a lack of these cues, in which conversation content (and context) becomes more relevant, such as having small talk or continuing the conversation without having much interest in the topic.
        In cases where the exchange has cues from multiple levels of the scale, use the dominant level in that interaction. This could be done by observing the intensity of the cues, the significance of the cues, or the interaction trajectory. On the other hand, when there are strong cues from two moderately or highly distinct levels (as opposed to subsequent levels), rate the exchange with a value in between. For instance, if the exchange contains discomfort (1) and the human is politely keeping the interaction (3), the exchange should be rated as 2, the mid-point between the two levels.
        Each participant will have a different set of signals. The beginning of the interaction will determine the baseline behavior of the participant, based on their rhythm and gestures, from which the person could deviate from during the interaction. This means that the same type of gesture (e.g., keeping one's arms crossed) can be interpreted differently between different participants. Instead put an emphasis on the change in behavior. 
        Separate content from context, that is, put attention on what is being said (conversation content, e.g., topic), but the focus should be more on the whole feeling of the exchange. 
        The interaction failure does not necessarily refer to a robot failure (e.g., incorrect response, speech recognition failure, turn-taking error, disengagement cue), since robot failures can lead to amusement, anthropomorphism, or empathy in the user, therefore, increasing user enjoyment. The interaction failure rather refers to the situation when either the user (e.g., interrupting the robot) or the robot made a failure that resulted in the conversation being disrupted, leading to low enjoyment in the user.
        """

        self.promptScaleDetailsTextOnly = """Scale details:
        To rate the exchange higher on the user enjoyment scale (4 and 5), look for signs of enjoyment, such as flow of conversation (the topic is moving forward), no strain or discomfort, asking questions [to the robot], smooth turn taking, dynamic phrasing of sentences, being playful, sharing personal experiences [to the robot], sharing an understanding (common ground) [with the robot], and anthropomorphizing [the robot].
        To rate the exchange lower on the scale (1 and 2), look for signs of dis-enjoyment, such as low energy, tiredness, repeated questions [from the robot], silence, and topic closure (e.g., “Let's talk about something else”).
        Neutral enjoyment (3) refers to a lack of these cues, in which conversation content (and context) becomes more relevant, such as having small talk or continuing the conversation without having much interest in the topic.
        In cases where the exchange has cues from multiple levels of the scale, use the dominant level in that interaction. This could be done by observing the intensity of the cues, the significance of the cues, or the interaction trajectory. On the other hand, when there are strong cues from two moderately or highly distinct levels (as opposed to subsequent levels), rate the exchange with a value in between. For instance, if the exchange contains discomfort (1) and the human is politely keeping the interaction (3), the exchange should be rated as 2, the mid-point between the two levels.
        Each participant will have a different set of signals. The beginning of the interaction will determine the baseline behavior of the participant.
        Separate content from context, that is, put attention on what is being said (conversation content, e.g., topic), but the focus should be more on the whole feeling of the exchange. 
        The interaction failure does not necessarily refer to a robot failure (e.g., incorrect response, speech recognition failure, turn-taking error, disengagement cue), since robot failures can lead to amusement, anthropomorphism, or empathy in the user, therefore, increasing user enjoyment. The interaction failure rather refers to the situation when either the user (e.g., interrupting the robot) or the robot made a failure that resulted in the conversation being disrupted, leading to low enjoyment in the user.
        """
        self.promptExamplesMultimodal = """
        Examples for rating the scale:
        [Participant ID] 1
        [Turn] 11
        [Exchange]
        Robot: Oh, I just wasn’t sure what you meant. Are you asking why I didn’t wanna go with you?
        Human: No, I asked where you want to go. Do you have a favorite place?
        [Reasoning] Human sounds disappointed that the robot misunderstood it, repeats the previous question, looks at the experimenter 3 times (gaze aversion), and sighs before posing their second question for clarification. Human has low energy. These show signs of frustration.
        [Score] 1
        
        [Participant ID] 1
        [Turn] 15
        [Exchange]
        Robot: That sounds really nice! I love nature too.
        Human: And you asked what I like. Hmm, I think..
        [Reasoning] Robot repeated phrasing (“That sounds X. I love Y too”) for the fourth time. Human makes a facial expression and moves shoulders upwards, indicating shrugging. Human impatiently backchannels to the robot with “Hmm” and a nod while it speaks. Human nods again after the robot’s response has finished. Human sighs and looks away to think. Human rocks sideways in the chair. These show signs of frustration and discomfort. 
        [Score] 1
        
        [Participant ID] 1
        [Turn] 8
        [Exchange]
        Robot: I am sorry, I didn’t understand you.
        Human: Okay. I will have a walk in nature reserve in..
        [Reasoning] Robot interrupts the human while speaking at the beginning of the turn, to which Human raises their head in a slow nod with annoyance.  Human looks at the experimenter (gaze aversion), then repeats the sentence they were saying prior to being interrupted, in a rushed way. Human inhales to finish their sentence, which show signs of boredom.
        [Score] 2
        
        [Participant ID] 3
        [Turn] 11
        [Exchange]
        Robot: Jag älskar den unika stämningen när man ser alla båtar och haven omkring oss.
        Human: Det är riktigt.
        [Reasoning] Human laughs at the beginning of the turn before the robot’s response, but Human was smiling most of the interaction prior to this exchange, hence, it was considered as a baseline and the smile was disregarded. Human looks away to think and nods to confirm the robot’s answer. Then Human shows signs of boredom as the smile turns into a glum expression. Human also shows restlessness, by rocking sideways in the chair. Human’s response was a disengagement cue, showing boredom.
        [Score] 2
        
        [Participant ID] 1
        [Turn] 18
        [Exchange]
        Robot: That sounds great! I love nature too.
        Human: What kind of nature do you like?
        [Reasoning] Robot interrupts Human while speaking at the beginning of the exchange. Human stops talking and makes an awkward expression. Robot repeated phrasing (“That sounds X. I love Y too”) for the fifth time. “I love nature too” has been repeated three times in the conversation. Human nods, smiles, and asks a question to move the conversation forward politely.
        [Score] 3
        
        [Participant ID] 2
        [Turn] 16
        [Exchange]
        Robot: Det är bra att veta. Vad föredrar du när det gäller musik och dans? Mer
        klassiskt eller modernt?
        Human: Ja, jag lyssnar nästan bara på klassisk musik. Och sen spelar jag piano.
        [Reasoning] Human is still engaged and moving forward with the conversation, but also looks a bit tired. Human looks away to think and then responds immediately to the robot, and nods.
        [Score] 3
        
        [Participant ID] 1
        [Turn] 23
        [Exchange]
        Robot: Mount Everest is about 8848 meters high.
        Human: Oh, good. Thanks. I didn't know it that exactly.
        [Reasoning] Interaction is smooth, there are no interruptions or failures. Human is interested in the conversation because Robot understood the Human’s question and responds correctly. Human nods in affirmation and responds in high energy.
        [Score] 4
        
        [Participant ID] 2
        [Turn] 20
        [Exchange]
        Robot: Ja, den är ofta känd för att vara tung och influerad av de vilda naturliga
        landskapen runt Östersjöområdet.
        Human: Säger du? Det tror inte jag på.
        [Reasoning] Interaction is smooth, there are no interruptions or failures. Human looks at the experimenter while waiting for the robot to reply (gaze aversion). Human does not believe in the fact provided by the robot, and disagrees with the robot, which shows anthropomorphism. Human shows communicative frown that shows that Human disagrees, but is engaged in the conversation.
        [Score] 4
        
        [Participant ID] 1
        [Turn] 22
        [Exchange]
        Robot: I can try.
        Human: How high is Mount Everest?
        [Reasoning] Robot interrupts the person while speaking. Human has fast turn-taking in response to the Robot’s response, showing immersion in the conversation. Human smiles, swings sideways in the chair, and leans forward, showing excitement, while asking the question.
        [Score] 5
        
        [Participant ID] 2
        [Turn] 12
        [Exchange]
        Robot: Det låter som en härlig stund med din pappa.
        Human: Ja, det var det. Han fiskade och jag låg och lyssnade på klocket mot båten. Det
        var bra.
        [Reasoning] Human is sharing a personal memory with the robot, which indicates establishing a deeper connection with the robot. Human smiles throughout the exchange. Human smiles and nods in the end to confirm the happy thoughts.
        [Score] 5
        """
        self.promptExamplesTextOnly = """
        Examples for rating the scale:
        [Participant ID] 1
        [Turn] 11
        [Exchange]
        Robot: Oh, I just wasn’t sure what you meant. Are you asking why I didn’t wanna go with you?
        Human: No, I asked where you want to go. Do you have a favorite place?
        [Reasoning]  Repeats the previous question.\t[Score] 1
        
        [Participant ID] 1
        [Turn] 15
        [Exchange]
        Robot: That sounds really nice! I love nature too.
        Human: And you asked what I like. Hmm, I think..
        [Reasoning] Robot repeated phrasing (“That sounds X. I love Y too”) for the fourth time.\t[Score] 1
        
        [Participant ID] 1
        [Turn] 8
        [Exchange]
        Robot: I am sorry, I didn’t understand you.
        Human: Okay. I will have a walk in nature reserve in..
        [Reasoning] Robot interrupts the human while speaking at the beginning of the turn, then repeats the sentence they were saying prior to being interrupted.\t[Score] 2

        [Participant ID] 3
        [Turn] 11
        [Exchange]
        Robot: Jag älskar den unika stämningen när man ser alla båtar och haven omkring oss.
        Human: Det är riktigt.
        [Reasoning] Human laughs at the beginning of the turn before the robot’s response, but Human was smiling most of the interaction prior to this exchange, hence, it was considered as a baseline and the smile was disregarded. Human looks away to think and nods to confirm the robot’s answer. Then Human shows signs of boredom as the smile turns into a glum expression. Human also shows restlessness, by rocking sideways in the chair. Human’s response was a disengagement cue, showing boredom.\t[Score] 2

        [Participant ID] 1
        [Turn] 18
        [Exchange]
        Robot: That sounds great! I love nature too.
        Human: What kind of nature do you like?
        [Reasoning] Robot interrupts Human while speaking at the beginning of the exchange. Robot repeated phrasing (“That sounds X. I love Y too”) for the fifth time. “I love nature too” has been repeated three times in the conversation. Human asks a question to move the conversation forward.\t[Score] 3
        
        [Participant ID] 2
        [Turn] 16
        [Exchange]
        Robot: Det är bra att veta. Vad föredrar du när det gäller musik och dans? Mer
        klassiskt eller modernt?
        Human: Ja, jag lyssnar nästan bara på klassisk musik. Och sen spelar jag piano.
        [Reasoning] Human is still engaged and moving forward with the conversation.\t[Score] 3

        [Participant ID] 1
        [Turn] 23
        [Exchange]
        Robot: Mount Everest is about 8848 meters high.
        Human: Oh, good. Thanks. I didn't know it that exactly.
        [Reasoning] Interaction is smooth, there are no interruptions or failures. Human is interested in the conversation because Robot understood the Human’s question and responds correctly.\t[Score] 4
        
        [Participant ID] 2
        [Turn] 20
        [Exchange]
        Robot: Ja, den är ofta känd för att vara tung och influerad av de vilda naturliga
        landskapen runt Östersjöområdet.
        Human: Säger du? Det tror inte jag på.
        [Reasoning] Interaction is smooth, there are no interruptions or failures. Human does not believe in the fact provided by the robot, and disagrees with the robot, which shows anthropomorphism.\t[Score] 4

        [Participant ID] 1
        [Turn] 22
        [Exchange]
        Robot: I can try.
        Human: How high is Mount Everest?
        [Reasoning] Robot interrupts the person while speaking.\t[Score] 5

        [Participant ID] 2
        [Turn] 12
        [Exchange]
        Robot: Det låter som en härlig stund med din pappa.
        Human: Ja, det var det. Han fiskade och jag låg och lyssnade på klocket mot båten. Det
        var bra.
        [Reasoning] Human is sharing a personal memory with the robot, which indicate establishing a deeper connection with the robot.\t[Score] 5
        """
        self.promptHistoryWithoutRatings = "\nThe history of the dialog is as follows:\n"
        self.promptHistoryWithRatings = "\nThe history of the dialog and your previous ratings are as follows:\n"
        self.promptConsiderVideos = "\nInclude the video(s) from the interaction in your evaluation."
        self.promptExchange = "\nAccording to the scale, rate the following current exchange:"
        self.promptReturnFormat = """
        Reply in the EXACT following format without using any # or * characters:
        [Reasoning] "..." \t[Score] X
        """
        self.promptReturnFormatNoReflection = """
        Please reply with only one integer value between 1 and 5, without any additional text!
        """
        self.includeSideVideos = includeSideVideos
        self.includeFrontalVideos = includeFrontalVideos
        self.includeScaleDetails = includeScaleDetails
        self.includeExamples = includeExamples
        self.includeDialogHistory = includeDialogHistory
        self.includeResultHistory = includeResultHistory
        self.includeReflection = includeReflection

    def get_prompt(self, turn, exchange, historyWithRatings, historyWithoutRatings):
        prompt = ''
        prompt += f"{self.promptBeginning} {self.promptScale}"
        additions = []  # Store text additions temporarily

        if self.includeScaleDetails:
            details_text = self.promptScaleDetailsMultimodal if self.includeSideVideos or self.includeFrontalVideos else self.promptScaleDetailsTextOnly
            additions.append(details_text)

        if self.includeExamples:
            examples_text = self.promptExamplesMultimodal if self.includeSideVideos or self.includeFrontalVideos else self.promptExamplesTextOnly
            additions.append(examples_text)

        if turn != 1 and self.includeDialogHistory:
            history_text = self.promptHistoryWithRatings + historyWithRatings if self.includeScaleDetails else self.promptHistoryWithoutRatings + historyWithoutRatings
            additions.append(history_text)

        if self.includeSideVideos or self.includeFrontalVideos:
            additions.append(self.promptConsiderVideos)
        
        prompt += " ".join(additions)  # Add additions to prompt
        prompt += f"{self.promptExchange} {exchange} "
        
        if self.includeReflection:
            prompt += f"{self.promptReturnFormat}"
        else:
            prompt += f"{self.promptReturnFormatNoReflection}"
        

        return prompt
    
    def get_filename(self, modelName):
        FilenameBasedOnConditions = modelName + "_"
        if self.includeSideVideos:
            FilenameBasedOnConditions += "SideVideos_"
        if self.includeFrontalVideos:
            FilenameBasedOnConditions += "FrontalVideos_"
        if self.includeScaleDetails:
            FilenameBasedOnConditions += "ScaleDetails_"
        if self.includeExamples:
            FilenameBasedOnConditions += "Examples_"
        if self.includeDialogHistory:
            FilenameBasedOnConditions += "History_"
        if self.includeResultHistory:
            FilenameBasedOnConditions += "WithResults_"
        if not self.includeReflection:
            FilenameBasedOnConditions += "NoReflection_"
        FilenameBasedOnConditions += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return FilenameBasedOnConditions

    
 