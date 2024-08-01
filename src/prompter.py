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
                return [cot, rewrite, response]
            else:
                rewrite = splits[0][9:]
                response = "\n".join(splits[1:])[10:]
                return [rewrite, response]
        except:
            return None


