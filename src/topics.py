import json
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Result:
    collection: str = None  # e.g. "AP"
    retrieval_model: str = None  # e.g. "ANCE"
    reranker: str = None  # e.g. "rankllama"
    metrics: Dict[str, float] = None  # e.g. {"ndcg": 0.5, "map": 0.6}

    def __eq__(self, another_instance: any) -> bool:
        if isinstance(another_instance, Result):
            return (
                self.collection == another_instance.collection 
                and 
                self.retrieval_model == another_instance.retrieval_model
                and
                self.reranker == another_instance.reranker
            )


@dataclass
class Reformulation:
    '''
    class for this structure:
    reformulation_name: "reformulated_description_by_[gpt-4-turbo]_using_[real_narrative_prompt]",

    reformulated_query: "The document discusses a pending antitrust case. To be relevant, a document will discuss a pending antitrust case and will identify the alleged violation as well as the government entity investigating the case. Identification of the industry and the companies involved is optional. The antitrust investigation must be a result of a complaint, NOT as part of a routine review.",

    results: [Result object1, Result object2, ...] 
    '''
    reformulation_name: str = None
    reformulated_query: str = None
    results: List[Result] = dataclasses.field(default_factory=list)

    
    def find_result(
        self, collection: str, 
        retrieval_model: str,
        reranker: str
        ) -> Result:
        '''
        Finds a result by its collection and retrieval model.

        Args:
            collection (str): The collection of the result to find.
            retrieval_model (str): The retrieval model of the result to find.
            reranker: The reranker of the result to find.

        Returns:
            Result: The found result object, or None if not found.
        '''
        dummy_result = Result(collection, retrieval_model, reranker)
        list_of_found_results = [result for result in self.results if result == dummy_result] 
        if len(list_of_found_results) == 0:
            return None
        elif len(list_of_found_results) == 1:
            return list_of_found_results[0]
        else:
            raise ValueError(f"Multiple results with the same collection [{collection}] and retrieval model [{retrieval_model}] as well as reranker [{reranker}] found in the reformulation object with name [{self.reformulation_name}]")
    
    def add_result(
            self, 
            collection:str, 
            retrieval_model:str, 
            reranker:str,
            result_dict: Dict) -> None:
        '''
        Adds a result to the reformulation.

        Args:
            result (Result): The result to add.
        '''
        result_found = self.find_result(collection, retrieval_model,reranker)
        if result_found is None:
            result = Result(collection, retrieval_model, result_dict,reranker)
            self.results.append(result)
        else:
            result_found.metrics = result_dict

    


@dataclass
class Turn:
    '''
    A class representing a turn.
    Attributes:
        turn_id (str): exmaple: "9-1-3" 
        conversation_id (str): example: "9-1"
        title (str): The title of the topic.
        current_utterance (str): The current turn utterance in the conversation.
        current_response (str): The current turn response in the conversation.
        response_provenance (List[str]): The provenance passage ids of the response.
        oracle_utterance (str): The resolved current turn utterance (manually) 
        context_uttrances (List[str]): The list of previous utterances in the conversation.
        ptkb (Dict[int, str]): The ptkb related to the conversation.
        ptkb_provenance (List[str]): The list of ptkb annotated related to the current turn  

    Methods:
        __init__: Initializes a new instance of the Turn class.
        __str__: Returns a string representation of the Turn object.
        __repr__: Returns a string representation of the Turn object.
        to_dict: Converts the Turn object to a dictionary.
        from_dict: Initializes the Turn object from a dictionary.
        find_reformulation: Finds a reformulation by its name.
        query_type_2_query: Generates/finds a query based on the specified query type.

    '''
    turn_id: str
    conversation_id: str 
    title: str
    current_utterance: str
    current_response: str
    response_provenance: List[str] = dataclasses.field(default_factory=list)
    oracle_utterance: str 
    context_uttrances: List[str] = dataclasses.field(default_factory=list) 
    ptkb: dict[int, str] = dataclasses.field(default_factory=dict)
    ptkb_provenance: List[str] = dataclasses.field(default_factory=list)

    # reformulations
    reformulations: List[Reformulation] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:

        print_str =  f"Turn ID: {self.turn_id}\n"
        print_str += f"Title: {self.title}\n"
        print_str += f"Current Utterance: {self.current_utterance}\n"
        print_str += f"Oracle Utterance: {self.oracle_utterance}\n"
        print_str += f"Number of reformulations: {len(self.reformulations)}\n"
        return print_str

    
    def __repr__(self) -> str:
        return self.__str__()
    

    def to_dict(self) -> Dict:
        '''
        Converts the Turn object to a dictionary.

        Returns:
            dict: The dictionary representation of the Turn object.
        '''

        turn_dict = {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "title": self.title,
            "current_utterance": self.current_utterance,
            "current_response": self.current_response,
            "response_provenance": self.response_provenance,
            "oracle_utterance": self.oracle_utterance,
            "context_uttrances": self.context_utterances,
            "ptkb": self.ptkb,
            "ptkb_provenance": self.ptkb_provenance
        }

        
        reformulations = []
        for reformulation in self.reformulations:
            reformulation_dict = {}
            if reformulation.reformulation_name is not None:
                reformulation_dict["reformulation_name"] = reformulation.reformulation_name
                reformulation_dict["reformulated_query"] = reformulation.reformulated_query
                reformulation_dict["results"] = []
                for result in reformulation.results:
                    result_dict = {}
                    if result.retrieval_model is not None:
                        result_dict["retrieval_model"] = result.retrieval_model
                    if result.metrics is not None:
                        result_dict["metrics"] = result.metrics
                    if result.reranker is not None:
                        result_dict["reranker"] = result.reranker
                    if result.collection is not None:
                        result_dict["collection"] = result.collection   
                    reformulation_dict["results"].append(result_dict)
            reformulations.append(reformulation_dict)
        
        turn_dict["reformulations"] = reformulations

        return turn_dict

    def from_ikat_topic_files(
        self, 
        ikat_topic_file: str
    ) -> None:
        
    def from_dict(self, turn_dict: Dict) -> None:
    
        '''
        Initializes the Turn object from a dictionary.

        Args:
            turn_dict (dict): The dictionary containing the turn information.
        '''
        self.turn_id = turn_dict["turn_id"]
        self.conversation_id = turn_dict["conversation_id"]
        self.title = turn_dict["title"]
        self.current_utterance = turn_dict["current_utterance"]
        self.current_response = turn_dict["current_response"]
        self.response_provenance = turn_dict["response_provenance"]
        self.oracle_utterance = turn_dict["oracle_utterance"]
        self.context_utterances = turn_dict["context_utterances"]
        self.ptkb = turn_dict["ptkb"]
        self.ptkb_provenance = turn_dict["ptkb_provenance"]

        # determine what are the reformulations
        reformulations = []

        for reformulation in turn_dict["reformulations"]:
            reformulation_obj = Reformulation(
                reformulation_name = reformulation["reformulation_name"],
                reformulated_query = reformulation["reformulated_query"],
                results=[]
            )

            for result_dict in reformulation["results"]:
                reformulation_obj.add_result(
                    collection = result_dict["collection"],
                    retrieval_model = result_dict["retrieval_model"],
                    reranker = result_dict["reranker"],
                    result_dict = result_dict["metrics"]
                )
            reformulations.append(reformulation_obj)


        self.reformulations = reformulations
    
    def find_reformulation(self, reformulation_name: str) -> Reformulation:
        '''
        Finds a reformulation by its name.

        Args:
            reformulation_name (str): The name of the reformulation to find.

        Returns:
            Reformulation: The found reformulation object, or None if not found.

        Raises:
            ValueError: If multiple reformulations with the same name are found.
        '''
        list_of_found_reformulations = [reformulation for reformulation in self.reformulations if reformulation.reformulation_name == reformulation_name]
        if len(list_of_found_reformulations) == 0:
            return None
        elif len(list_of_found_reformulations) == 1:
            return list_of_found_reformulations[0]
        else:
            raise ValueError(f"Multiple reformulations with the same name {reformulation_name} found in the turn object with id {self.turn_id}")
    

    def query_type_2_query(self, args):
        '''
        Generates a query based on the specified query type.

        Args:
            args: The arguments containing the query type and other parameters.

        Returns:
            str: The generated/found query.

        Raises:
            ValueError: If the specified query type is not supported.
        '''
        query_type = args.original_query_type
        run_reformulate = args.run_reformulate
        final_query = ""

        if query_type == "title":
            final_query = self.topic
        elif query_type == "description":
            final_query = self.description
        elif query_type == "narrative":
            final_query = self.narrative
        elif query_type == "title+description":
            final_query = self.topic + ", " + self.description
        elif query_type == "title+narrative":
            final_query = self.topic + ", " + self.narrative
        elif query_type == "description+narrative":
            final_query = self.description + ", " + self.narrative
        elif query_type == "title+description+narrative":
            final_query = self.topic + ", " + self.description + ", " + self.narrative
        elif query_type == "reformulation":
            args.query_type = f'reformulated_description_by_[{args.rewrite_model}]_using_[{args.prompt_type}]'
        elif query_type == "pseudo_narrative":
            args.query_type = f'pseudo_narrative_by_[{args.rewrite_model}]_using_[{args.prompt_type}]'
        else:
            raise ValueError(f"query_type {query_type} not supported")

        # check if we have already added this version to the json file
        reformulation = self.find_reformulation(args.query_type)

        if reformulation is not None:
            final_query = reformulation.reformulated_query
        else:
            if run_reformulate:
                ## TODO, if we want to rewrite the query, we should do it here.
                # final_qeury = rewrite....
                pass
            self.reformulations.append(
                Reformulation(
                    reformulated_query = final_query,
                    reformulation_name = args.query_type,
                    results = []
                    )
                )

        return final_query


def save_turns_to_json(
        turns: List[Turn],
        output_turn_path: str
        ) -> List[Dict]:
    '''
    Save a list of Turn objects to a json file
    '''
    turn_list = [turn.to_dict() for turn in turns]
    
    with open(output_turn_path, 'w') as f:
        json.dump(turn, f, indent=4)
    
    return turn_list

        
def load_turns_from_json(
        input_turn_path: str, 
        range_start: int = 0,
        range_end: int = 150,
        ) -> List[Turn]:

    ### read json file
    with open(input_turn_path, 'r') as f:
        turns = json.load(f)
    
    ### resulting list of Turn objects
    turn_objects = []
    
    for turn in turns[range_start:range_end]:
        turn_object = Turn()
        turn_object.from_dict(turn) 
        turn_objects.append(turn_object)
    
    return turn_objects
    


                 
                
