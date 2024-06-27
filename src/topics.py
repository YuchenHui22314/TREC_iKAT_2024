import json
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict 
from constants import IKAT_23_EVALUATED_TURNS
from tqdm import tqdm


@dataclass
class Result:
    collection: str = None  # e.g. "AP"
    retrieval_model: str = None  # e.g. "ANCE"
    reranker: str = None  # e.g. "rankllama"
    retrieval_query_type: str = None # e.g. what is UdeM?
    reranking_query_type: str = None # e.g. what is UdeM? Specifically, what is the history of UdeM?
    generation_query_type: str = None # e.g. please provide me with information about UdeM
    metrics: Dict[str, float] = None  # e.g. {"ndcg": 0.5, "map": 0.6}

    def __eq__(self, another_instance: any) -> bool:
        if isinstance(another_instance, Result):
            return (
                self.collection == another_instance.collection 
                and 
                self.retrieval_model == another_instance.retrieval_model
                and
                self.reranker == another_instance.reranker
                and 
                self.retrieval_query_type == another_instance.retrieval_query_type
                and
                self.reranking_query_type == another_instance.reranking_query_type
                and
                self.generation_query_type == another_instance.generation_query_type
            )


@dataclass
class Reformulation:
    '''
    class for this structure:
    reformulation_name: "reformulated_description_by_[gpt-4-turbo]_using_[real_narrative_prompt]",

    reformulated_query: "The document discusses a pending antitrust case. To be relevant, a document will discuss a pending antitrust case and will identify the alleged violation as well as the government entity investigating the case. Identification of the industry and the companies involved is optional. The antitrust investigation must be a result of a complaint, NOT as part of a routine review.",

    '''
    reformulation_name: str = None
    reformulated_query: str = None

    
    
    


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
    turn_id: str = None
    conversation_id: str = None
    title: str = None
    current_utterance: str = None
    current_response: str = None
    oracle_utterance: str = None
    response_provenance: List[str] = field(default_factory=list)
    context_utterances: List[str] = field(default_factory=list) 
    ptkb: dict[int, str] = field(default_factory=dict)
    ptkb_provenance: List[str] = field(default_factory=list)

    # reformulations
    reformulations: List[Reformulation] = field(default_factory=list)

    # results:
    results: List[Result] = field(default_factory=list)

    def __str__(self) -> str:

        print_str =  f"Turn ID: {self.turn_id}\n"
        print_str += f"Title: {self.title}\n"
        print_str += f"Current Utterance: {self.current_utterance}\n"
        print_str += f"Oracle Utterance: {self.oracle_utterance}\n"
        print_str += f"Number of reformulations: {len(self.reformulations)}\n"
        print_str += f"Number of results: {len(self.results)}\n"
        return print_str

    
    def __repr__(self) -> str:
        return self.__str__()
    
    def find_result(
        self, 
        collection: str, 
        retrieval_model: str,
        reranker: str,
        retrieval_query_type: str,
        reranking_query_type: str,
        generation_query_type: str
        ) -> Result:
        '''
        Finds a result by its collection and retrieval model, reranker, retrieval query type, reranking query type, and generation query type.

        Args:
            collection (str): The collection of the result to find.
            retrieval_model (str): The retrieval model of the result to find.
            reranker: The reranker of the result to find.
            retrieval_query_type: The retrieval query type of the result to find.
            reranking_query_type: The reranking query type of the result to find.
            generation_query_type: The generation query type of the result to find.

        Returns:
            Result: The found result object, or None if not found.
        '''
        dummy_result = Result(
            collection, 
            retrieval_model, 
            reranker,
            retrieval_query_type,
            reranking_query_type,
            generation_query_type
            )
        list_of_found_results = [result for result in self.results if result == dummy_result] 
        if len(list_of_found_results) == 0:
            return None
        elif len(list_of_found_results) == 1:
            return list_of_found_results[0]
        else:
            raise ValueError(f"Multiple results with the same collection [{collection}], retrieval model [{retrieval_model}] reranker [{reranker}], retrieval query type [{retrieval_query_type}], reranking query type [{reranking_query_type}], generation query type [{generation_query_type}] found in the Turn object with id [{self.turn_id}]")
    
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
    
    def add_result(
            self, 
            collection:str, 
            retrieval_model:str, 
            reranker:str,
            retrieval_query_type:str,
            reranking_query_type:str,
            generation_query_type:str,
            metrics_dict: Dict
            ) -> None:
        '''
        Adds a result to the Turn.

        Args:
            result (Result): The result to add.
        '''
        result_found = self.find_result(
            collection, 
            retrieval_model,
            reranker,
            retrieval_query_type,
            reranking_query_type,
            generation_query_type
            )
        if result_found is None:
            result = Result(
            collection, 
            retrieval_model,
            reranker,
            retrieval_query_type,
            reranking_query_type,
            generation_query_type,
            metrics_dict
            )
            self.results.append(result)
        else:
            result_found.metrics = metrics_dict
        

    def add_reformulation(
            self,
            reformulation_name: str,
            reformulated_query: str
    ) -> None:
        '''
        Adds a reformulation to the turn.

        Args:
            reformulation (Reformulation): The reformulation to add.
        '''
        reformulation_found = self.find_reformulation(reformulation_name)
        if reformulation_found is None:
            self.reformulations.append(
            Reformulation(
                reformulated_query = reformulated_query,
                reformulation_name = reformulation_name,
                )
            )


    def query_type_2_query(
        self, 
        original_query_type: str, 
        ) -> str:
        '''
        Generates a query based on the specified query type.

        Args:
            original_query_type (str): The query type to generate the query for.
            run_reformulate (bool): Whether to run reformulation or not.

        Returns:
            str: The generated/found query.

        Raises:
            ValueError: If the specified query type is not supported.
        '''
        query_type = original_query_type
        final_query = ""

        if query_type == "current_utterance":
            final_query = self.context_utterances
        elif query_type == "oracle_utterance":
            final_query = self.oracle_utterance
        else:
            raise ValueError(f"query_type {query_type} not supported")

        # check if we have already added this version to the json file
        reformulation = self.find_reformulation(query_type)

        if reformulation is not None:
            final_query = reformulation.reformulated_query
        else:
            self.reformulations.append(
                Reformulation(
                    reformulated_query = final_query,
                    reformulation_name = query_type,
                    )
                )

        return final_query
        
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


        for reformulation in turn_dict["reformulations"]:
            self.add_reformulation(
                reformulation_name = reformulation["reformulation_name"],
                reformulated_query = reformulation["reformulated_query"],
            )

        for result_dict in turn_dict["results"]:
            self.add_result(
                collection = result_dict["collection"],
                retrieval_model = result_dict["retrieval_model"],
                reranker = result_dict["reranker"],
                retrieval_query_type = result_dict["retrieval_query_type"],
                reranking_query_type = result_dict["reranking_query_type"],
                generation_query_type = result_dict["generation_query_type"],
                metrics_dict = result_dict["metrics"]
            )
        


    
    
def save_turns_to_json(
        turns: List[Turn],
        output_turn_path: str
        ) -> List[Dict]:
    '''
    Save a list of Turn objects to a json file
    '''
    turn_list = [asdict(turn) for turn in turns]
    
    with open(output_turn_path, 'w') as f:
        json.dump(turn_list, f, indent=4)
    
    return turn_list

            
def load_turns_from_json(
        input_topic_path: str, 
        range_start: int = 0,
        range_end: int = -1,
        ) -> List[Turn]:

    ### read json file
    with open(input_topic_path, 'r') as f:
        turns = json.load(f)
    
    ### resulting list of Turn objects
    turn_objects = []
    
    if range_end == -1:
        turns = turns[range_start:]
    else:
        turns = turns[range_start : range_end + 1]

    for turn in turns:
        turn_object = Turn()
        turn_object.from_dict(turn) 
        turn_objects.append(turn_object)
    
    return turn_objects
    

def load_turns_from_ikat_topic_files(
    ikat_topic_file: str
) -> List[Turn]:
    '''
    Load a list of Turn objects from the iKAT topic files. Source: preprocess_cast23.py by Fengran 
    '''
    with open(ikat_topic_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    
    list_of_turns = []

    # Iterate through each item in the original data
    for item in data:
        queries = []
        number = item["number"]
        title = item["title"]
        ptkb = item["ptkb"]
        for turn in item["turns"]:
            if turn['turn_id'] == '1':
                queries = []
            turn_id = f"{number}-{turn['turn_id']}"
            queries.append(turn["utterance"])
            turn_object = Turn(
                turn_id = turn_id,
                conversation_id = number,
                title = title,
                current_utterance = turn["utterance"],
                current_response = turn["response"],
                response_provenance = turn["response_provenance"],
                oracle_utterance = turn["resolved_utterance"],
                context_utterances = queries[:-1],
                ptkb = ptkb,
                ptkb_provenance = turn["ptkb_provenance"]
            )
            list_of_turns.append(turn_object)
    
    return list_of_turns

def filter_ikat_23_evaluated_turns(
    turns: List[Turn]
    ) -> List[Turn]:

        '''
        Filter the turns to only include the evaluated turns in iKAT 23
        '''

        filtered_turns = []
        for turn in turns:
            if turn.turn_id in IKAT_23_EVALUATED_TURNS:
                filtered_turns.append(turn)
        return filtered_turns

    



                 
                

                 
                
