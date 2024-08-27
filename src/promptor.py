import re
import json

def check_length(prompt, max_length):
    n = len(prompt.split(' '))
    if n >= max_length:
        return False
    return True

class RewriteAndResponsePromptor:
    def __init__(
        self, 
        demo_file, 
        enable_cot=False) -> None:    
        
        self.instruction = "For an information-seeking dialog, please help reformulate the question into rewrite that can fully express the user's information needs without the need of context, but also generate an informative response to answer the question."
        self.enable_cot = enable_cot
        self.demo = self.get_demo(demo_file)
        if self.demo != "":
            self.instruction += " I will give you several example multi-turn dialogs, where each turn contains a question as well as a rewrite and a response that you need to generate."
            if enable_cot:
                self.instruction += " The rewrite part begins with a sentence explaining the reason for the generated rewrite."
        if enable_cot:
            self.tail_instruction = "Now, you should give me the rewrite and response of the **Current Question** under the **Context**. The output format should always be: \"Rewrite: $Reason. So the question should be rewritten as: $Rewrite\nResponse: $Response.\" Note that you should always try to rewrite it and generate an informative response. Never ask for clarification or say you don't understand it in the generated rewrite and response. Go ahead!"
        else:
            self.tail_instruction = "Now, you should give me the rewrite and response of the **Current Question** under the **Context**. The output format should always be:\nRewrite: $Rewrite\nResponse: $Response.\nNote that you should always try to rewrite it and generate an informative response. Never ask for clarification or say you don't understand it in the generated rewrite and response. Go ahead!"
        self.stop_tokens = None
                            
    
    def get_demo(self, demo_file):
        try:
            with open(demo_file, "r") as f:
                demos = json.load(f)
        except:
            print("warning: No demonstration file.")
            return ""
        
        examples = []
        for demo in demos:
            turns = demo['turns']
            
            dialog = []
            for turn in turns:
                if self.enable_cot:
                    rewrite = turn['cot'] + " So the question should be rewritten as: {}".format(turn['manual_rewrite'])
                else:
                    rewrite = turn['manual_rewrite']
                turn_text = "Question: {}\nRewrite: {}\nResponse: {}".format(turn['question'], rewrite, turn['response'])         
                dialog.append(turn_text)
            dialog = "\n\n".join(dialog)
            
            examples.append(dialog)
        
        for i in range(len(examples)):
            examples[i] = "Example #{}:\n".format(i+1) + examples[i]
        
        return "\n\n".join(examples)
    def build_turn_prompt(self, context, current_turn):
        # context
        this_dialog = []
        if not context:
            this_dialog.append("N/A")
        else:
            for turn in context:
                this_dialog.append("Question: {}\nResponse: {}".format(turn.current_utterance, turn.current_response))
        
        this_dialog[0] = "Context:\n" + this_dialog[0]
            
        # current turn
        this_dialog.append("Current Question: " + current_turn.current_utterance)
        this_dialog = "\n\n".join(this_dialog)  
        this_dialog = "YOUR TASK (only questions and responses may be given):\n" + this_dialog
        this_prompt = [self.instruction, self.demo, this_dialog, self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    

    def parse_returned_text(self, text):
        text = text.strip()
        try:
            splits = text.split('\n')
            if splits[0][:9] != "Rewrite: " or splits[1][:10] != "Response: ":
                return None
            if self.enable_cot:
                rewrite_text = splits[0][9:]
                fixed_sentence = "So the question should be rewritten as: "
                index = rewrite_text.find(fixed_sentence)
                if index != -1:
                    cot = rewrite_text[:index]
                    rewrite = rewrite_text[index + len(fixed_sentence):]
                else:
                    return None            
                
                response = "\n".join(splits[1:])[10:]
                return [rewrite, response, cot]
            else:
                rewrite = splits[0][9:]
                response = "\n".join(splits[1:])[10:]
                return [rewrite, response]
        except:
            return None

class RARPersonalizedCoTPromptor:
    def __init__(
        self, 
        demo_file, 
        enable_cot=False,
        zero_shot_cot=False,
        one_shot_cot=False,
        cot_format = "cot_seperate") -> None:
        
        self.enable_cot = enable_cot
        self.zero_shot_cot = zero_shot_cot
        self.one_shot_cot = one_shot_cot
        self.demo = self.get_demo(demo_file, cot_format)

        if self.enable_cot:
            cot_1 = "\n\t(2) Provide your reasoning process in terms of how to adopt de-contextualizaiton (a. & b.) as well as personalization (c.) before rewriting the question. \n\t(3) "
            cot_2 = "Please also provide your reasoning that justifies the way you rewrite the query. The style of the reasoning should be similar to those given in the examples. "
            cot_3 = "Reason: $Reason\n"
            if cot_format == "cot_seperate":
                cot_4 = "So the question should be rewritten as:\n" 
            elif cot_format == "cot_rewrite_together":
                cot_4 = ""
        else:
            cot_1 = "\n\t(2) "
            cot_2 = ""
            cot_3 = ""
            cot_4 = ""


        


        # head_instruction
        self.instruction = f"# Task Description:\nYou will be given\n\t(1) An information-seeking dialog between an user and an intelligent assistant.\n\t(2) The profile of the user, in form of several sentences describing his/her background information.\nYour tasks are as follows:\n\t(1) Help the assistant rewrite the user's question such that:\n\t\ta. The rewritten question can fully express the user's information needs without the need of dialog context. \n\t\tb. The assistant could use the rewritten question as a search engine query to gather supporting documents that can help answer the user's question.\n\t\tc. Please analyze the quesiton's nature, and judge if it is necessary to personalize. If so, please add personalized elements to the question based on the user's profile.{cot_1}Provide an informative response to the question."

        if self.enable_cot:
            self.instruction += "\n\nNow, I will give you several example multi-turn dialogs with their user profiles, where each turn contains a question, a rewrite, as well as a response by the intelligent assistant."
            if one_shot_cot:
                self.instruction += " The reasoning explaining the de-contextualizaiton and personalization consideration while rewriting the question is also provided before the rewrite part."

        self.tail_instruction = f"Now, please provide the rewrite and the response for the **Last Question** under the **Dialog Context**, considering the **User Profile**. {cot_2}The output format should always be:\n\n{cot_3}{cot_4}Rewrite: $Rewrite\nResponse: $Response \n\nGo ahead!"

        self.stop_tokens = None
                            
    def get_demo(self, demo_file, cot_format):
        try:
            with open(demo_file, "r") as f:
                demos = json.load(f)
        except:
            print("warning: No demonstration file.")
            return ""
        
        examples = []
        for demo in demos:
            turns = demo['turns']
            ptkb_dict = demo['ptkb']

            # ptkb
            ptkb_instruction = []
            ptkb_instruction.append("Example user profile:")
            for num, ptkb_sentence in ptkb_dict.items():
                ptkb_instruction.append("{}. {}".format(num, ptkb_sentence))
            
            ptkb_instruction = "\n".join(ptkb_instruction)

            # conversation turns
            dialog = []
            for i, turn in enumerate(turns):
                question = turn['question']
                rewrite = turn['manual_rewrite']
                response = turn['response']

                turn_text = ""
                if self.one_shot_cot:
                    cot = turn['cot']

                    if cot_format == "cot_seperate":
                        turn_text = f"Question {i+1}: {question}\nReason {i+1}: {cot}\nSo the question should be rewritten as:\nRewrite {i+1}: {rewrite}\nResponse {i+1}: {response}"


                    elif cot_format == "cot_rewrite_together":
                        rewrite = cot + " So the question should be rewritten as: " + rewrite 
                        turn_text = f"Question {i+1}: {question}\nRewrite {i+1}: {rewrite}\nResponse {i+1}: {response}"

                else:
                    rewrite = turn['manual_rewrite']
                    turn_text = f"Question {i+1}: {question}\nRewrite {i+1}: {rewrite}\nResponse {i+1}: {response}"

                dialog.append(turn_text)

            dialog = "\n\n".join(dialog)

            # add ptkb before dialog
            dialog = ptkb_instruction + "\n\nExample dialog:\n" + dialog
            
            examples.append(dialog)
        
        for i in range(len(examples)):
            examples[i] = "Example ########### {} ##########\n".format(i+1) + examples[i] + "\n######################"
        
        return "\n\n".join(examples)
    
    
    def build_turn_prompt(self, context, ptkb_dict, current_turn):
        # ptkb
        ptkb_instruction = []
        ptkb_instruction.append("Here is the **User Profile**:\n")
        for num, ptkb_sentence in ptkb_dict.items():
            ptkb_instruction.append("{}. {}".format(num, ptkb_sentence))
        
        ptkb_instruction = "\n".join(ptkb_instruction)


        # previous turn context
        this_dialog = []
        if not context:
            this_dialog.append("N/A (this is the first question in the dialog, so no previous dialog context)")
        else:
            for i, turn in enumerate(context):
                this_dialog.append(f"Question {i+1}: {turn.current_utterance}\nResponse {i+1}: {turn.current_response}")
        
        this_dialog[0] = "Here is the **Dialog Context**:\n\n" + this_dialog[0]
            
        # current turn
        this_dialog.append("**Last Question**: " + current_turn.current_utterance)
        this_dialog = "\n\n".join(this_dialog)  
        
        # combine to form the prompt
        this_prompt = []
        this_prompt.append(self.instruction)
        this_prompt.append(self.demo)
        this_prompt.append("# Now, the exmamples are over. Let's move to the dialog and the user profile you have to consider.")
        this_prompt.append(ptkb_instruction)
        this_prompt.append(this_dialog)
        this_prompt.append(self.tail_instruction)
        
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    

    def parse_returned_text(self, text):
        text = text.strip()
        try:
            splits = text.split('\n')
            rewrite = None
            response = None
            cot = None

            for line in splits:
                if line[:8] == "Rewrite:":
                    rewrite = line[8:].strip()
                elif line[:9] == "Response:":
                    response = line[9:].strip()
                elif line[:7] == "Reason:":
                    cot = line[7:].strip()
                
            if rewrite == None or response == None:
                return None 
            if self.enable_cot and cot == None:
                return None

            return [rewrite, response, cot]
        except Exception as e:
            print(e)
            return None
    
class PersonalizeViaPTKBSummaryPrompter:
    def __init__(
        self,
        enable_cot = False 
        ) -> None:    

        self.enable_cot = enable_cot
        
        self.instruction = f"You will receive a **Profile Summary** of a search engine user, as well as the **User's Question** presented to the search engine. Your task is to:\n\t1. Reformulate the query by incorporating relevant information from the user's profile to create a personalized query.\n\t2. Provide a personalized response to the reformulated query, ensuring it is tailored to the user's specific needs and context."

        if self.enable_cot:
            cot_instruction1 = "Please also provide your reasoning which explains how you incorporated the user's profile information to get the rewrite and response. "
            cot_instruction2 = "Reason: $Reason\n"
        else:
            cot_instruction1 = ""
            cot_instruction2 = ""

        self.tail_instruction = f"Now, please provide the personalized reformulated user question and the personalized response. {cot_instruction1}The output format should always be:\n\n{cot_instruction2}Rewrite: $Rewrite\nResponse: $Response\n\nGo ahead!"
        

    def build_turn_prompt(self, summary, user_query):

        summary_instruction = f"Here is the user's **Profile Summary**:\n{summary}"
        user_query_instruction = f"Here is the **User's Question**:\n{user_query}"
        
        this_prompt = [self.instruction, summary_instruction, user_query_instruction, self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt

    def parse_returned_text(self, text):
        text = text.strip()
        try:
            splits = text.split('\n')
            rewrite = None
            response = None
            cot = None

            for line in splits:
                if line[:8] == "Rewrite:":
                    rewrite = line[8:].strip()
                elif line[:9] == "Response:":
                    response = line[9:].strip()
                elif line[:7] == "Reason:":
                    cot = line[7:].strip()
                
            if rewrite == None or response == None:
                return None 
            if self.enable_cot and cot == None:
                return None

            return [rewrite, response, cot]
        except Exception as e:
            print(e)
            return None

class SummarizePTKBPromptor:
    def __init__(
        self, 
        ) -> None:    
        
        self.instruction = f"You will be given a persona of a search engine user, in form of several sentences describing his/her background information. Please provide a summary of the persona. The summary should cover all the key points and main ideas presented in the original persona, while also condensing the information into a concise and easy-to-understand format. The length of the summary should be appropriate to capture the main points and key details of the text, without including unnecessary information or becoming overly long."
        self.tail_instruction = "Now, please provide a summary of the **User Persona**. The summary should be concise and easy to understand, while also covering all the key points and main ideas presented in the original persona. The output format should always be:\n\nSummary: $Summary\n\nGo ahead!"
    
    def build_turn_prompt(self, ptkb_dict):

        ptkb_instruction = []
        ptkb_instruction.append("Here is the **User Persona**:\n")
        for num, ptkb_sentence in ptkb_dict.items():
            ptkb_instruction.append("{}. {}".format(num, ptkb_sentence))
        
        ptkb_instruction = "\n".join(ptkb_instruction)

        this_prompt = [self.instruction, ptkb_instruction,  self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    
    def parse_returned_text(self, text):
        if text[:9] != "Summary: ":
            return None
        else:
            return text[9:]
        
class PersonalizedCIRQueryExpansionPromptor:
    def __init__(
        self, 
        demo_file, 
        enable_personalization=False,
        enable_demo=False,
        enable_cot=False,
        enable_relevance_explanation=False,##TODO
        enable_oracle=False,##TODO
        enable_relevance_feedback=False##TODO
        ) -> None:    
        
        self.enable_demo = enable_demo
        self.enable_cot = enable_cot
        self.enable_relevance_explanation = enable_relevance_explanation
        self.enable_oracle = enable_oracle
        self.enable_relevance_feedback = enable_relevance_feedback
        self.enable_personalization = enable_personalization

        ######################
        # 1. instruction
        ######################
        if enable_personalization:
            personalization_1 = ", as well as the persona of the user"
            personalization_2 =" and user persona" 
            personalization_3 =" personalized" 
            personalization_4 = " and the **User Persona**"
        else:
            personalization_1= ""
            personalization_2= ""
            personalization_3= ""
            personalization_4= ""

        if enable_cot:
            cot_1 = f"Please also provide your reasoning which explains why you suggest this expansion term based on the provided dialog {personalization_2}. The output format should always be:" 
            cot_2 = "Reason: $Reason\n"
        else:
            cot_1 = "Please do not provide any reasoning, simply yield solely the expansion term in the following format:"
            cot_2 = ""

        # head_instruction
        self.instruction = \
        f"You will be given an information-seeking dialog between an user and a system{personalization_1}. Your tasks are as follows:\n\n\t1. Infer the user's underlying information need expressed by the last question, with the aid of the provided dialog{personalization_2}.\n\t2. Suggest an expansion term to add in the last question, which corresponds to the keyword that has the highest probability to appear in the set of all possible relevant documents. Please suggest based on your understanding of the user's{personalization_3} information need and what should a relevant document contain."


        # tail_instruction
        self.tail_instruction = f"Now, please suggest the expansion term that has the highest probability to appear in the documents set relevant to **Last Question** under the **Dialog Context**{personalization_4}. {cot_1} \n\n{cot_2}Keyword: $Keyword\n\nGo ahead!"

        ######################
        # 2. demonstration
        ######################
        if self.enable_demo:
            self.demo = self.get_demo(demo_file)
            if self.demo != "":
                self.instruction += " Here are several example multi-turn dialogs, where each turn contains a question as well as a rewrite and a response that you need to generate."
                if enable_cot:
                    self.instruction += " The rewrite part begins with a sentence explaining the reason for the generated rewrite."

            self.stop_tokens = None
        else:
            self.demo = ""

        ############################
        # 3. relevance explanation
        ############################

        self.relevance_explanation = "Please note that in this context, 'relevance' refers to documents that contain information related to the user's information need. This is distinct from authenticity, as a relevant document does not necessarily have to be accurate or correct. The key criterion is that the document's topic should align with the user's search intent. For example, both 'The capital of France is Beijing' and 'The capital of France is Paris' would be considered equally relevant, even though only the latter is factually correct."
        
        

                            
    def get_demo(self, demo_file, format):
        try:
            with open(demo_file, "r") as f:
                demos = json.load(f)
        except:
            print("warning: No demonstration file.")
            return ""

        
        examples = []
        for demo in demos:
            turns = demo['turns']
            
            dialog = []
            for turn in turns:
                
                if self.enable_cot:
                    rewrite = turn['cot'] + " So the question should be rewritten as: {}".format(turn['manual_rewrite'])
                else:
                    rewrite = turn['manual_rewrite']
                turn_text = "Question: {}\nRewrite: {}\nResponse: {}".format(turn['question'], rewrite, turn['response'])         

                dialog.append(turn_text)
            dialog = "\n\n".join(dialog)
            
            examples.append(dialog)
        
        for i in range(len(examples)):
            examples[i] = "Example #{}:\n".format(i+1) + examples[i]
        
        return "\n\n".join(examples)
    
    def build_turn_prompt(self, context, ptkb_dict, current_turn):
        # ptkb
        ptkb_instruction = []
        ptkb_instruction.append("Here is the **User Persona**:\n")
        for num, ptkb_sentence in ptkb_dict.items():
            ptkb_instruction.append("{}. {}".format(num, ptkb_sentence))
        
        ptkb_instruction = "\n".join(ptkb_instruction)


        # previous turn context
        this_dialog = []
        if not context:
            this_dialog.append("N/A (this is the first question in the dialog, so no previous dialog context)")
        else:
            for turn in context:
                this_dialog.append("Question: {}\nResponse: {}".format(turn.current_utterance, turn.current_response))
        
        this_dialog[0] = "Here is the **Dialog Context**:\n\n" + this_dialog[0]
            
        # current turn
        this_dialog.append("**Last Question**: " + current_turn.current_utterance)
        this_dialog = "\n\n".join(this_dialog)  
        
        # combine to form the prompt
        this_prompt = []
        this_prompt.append(self.instruction)
        if self.enable_relevance_explanation:
            this_prompt.append(self.relevance_explanation)
        if self.enable_demo:
            this_prompt.append(self.demo)
        if self.enable_personalization:
            this_prompt.append(ptkb_instruction)
        this_prompt.append(this_dialog)
        this_prompt.append(self.tail_instruction)
        
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    


    def parse_returned_text(self, text):
        text = text.strip()
        try:
            splits = text.split('\n')
            if splits[-1][:7] != "Reason:":
                return None
            if self.enable_cot:
                rewrite_text = splits[0][9:]
                fixed_sentence = "So the question should be rewritten as: "
                index = rewrite_text.find(fixed_sentence)
                if index != -1:
                    cot = rewrite_text[:index]
                    rewrite = rewrite_text[index + len(fixed_sentence):]
                else:
                    return None            
                
                response = "\n".join(splits[1:])[10:]
                return [rewrite, response, cot]
            else:
                rewrite = splits[0][9:]
                response = "\n".join(splits[1:])[10:]
                return [rewrite, response]
        except:
            return None
