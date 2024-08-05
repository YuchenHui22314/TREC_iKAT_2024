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

        # head_instruction
        self.instruction = \
        '''You will be given an information-seeking dialog between an user and a system, as well as the persona of the user, in form of several sentences describing the user's background information. Your task is to
        
        1. Infer the user's underlying information need expressed by the last question, with the aid of the provided dialog and user persona.
        2. Suppose that R represents the set of all possible documents relevant to user's query, please propose the keyword that has the highest probability to appear in the relevant documents set as an expansion term to add in the last question, based on your understanding of the user's information need and what should a relevant document contain.'''


        # tail_instruction
        if enable_cot:
            self.tail_instruction = "Now, you should give me the expansion term that has the highest probability to appear in the documents set relevant to **Last Question** under the **Dialog Context**. The output format should always be:\n\nReason: $Reason\nKeyword: $Keyword\n\nNote that you should never ask for clarification or say you don't understand it in the generated rewrite and response. Go ahead!"
        else:
            self.tail_instruction = "Now, you should give me the expansion term that has the highest probability to appear in the documents set relevant to **Last Question** under the **Dialog Context**. Please do not provide your reasoning, just yield solely the expansion term in the following format: \n\nKeyword: $Keyword\n\nGo ahead!"

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
        if self.enable_demo:
            this_prompt.append(self.demo)
        if self.enable_personalization:
            this_prompt.append(ptkb_instruction)
        this_prompt.append(this_dialog)
        this_prompt.append(self.tail_instruction)
        
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    


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
