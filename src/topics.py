import json
from typing import List, Dict

class Result:
    def __init__(self,
        collection: str = None, # e.g. "AP"
        retrieval_model: str = None, # e.g. "ANCE"
        metrics: Dict[str, float] = None # e.g. {"ndcg": 0.5, "map": 0.6}
        ) -> None:

        self.collection = collection
        self.retrieval_model = retrieval_model
        self.metrics = metrics
    
    def __eq__(self, another_instance: any) -> bool:
        if isinstance(another_instance, Result):
            return (
                self.collection == another_instance.collection 
                and 
                self.retrieval_model == another_instance.retrieval_model
                )


class Reformulation:
    '''
    class for this structure:
    reformulation_name: "reformulated_description_by_[gpt-4-turbo]_using_[real_narrative_prompt]",

    reformulated_query: "The document discusses a pending antitrust case. To be relevant, a document will discuss a pending antitrust case and will identify the alleged violation as well as the government entity investigating the case. Identification of the industry and the companies involved is optional. The antitrust investigation must be a result of a complaint, NOT as part of a routine review.",

    results: [Result object1, Result object2, ...] 
    '''
    
    def __init__(self,
                 reformulation_name: str = None,
                 reformulated_query: str = None,
                 results: List[Result] = []
                 ) -> None:

        self.reformulation_name = reformulation_name
        self.reformulated_query = reformulated_query
        self.results = results
    
    def find_result(self, collection: str, retrieval_model: str) -> Result:
        '''
        Finds a result by its collection and retrieval model.

        Args:
            collection (str): The collection of the result to find.
            retrieval_model (str): The retrieval model of the result to find.

        Returns:
            Result: The found result object, or None if not found.
        '''
        dummy_result = Result(collection, retrieval_model)
        list_of_found_results = [result for result in self.results if result == dummy_result] 
        if len(list_of_found_results) == 0:
            return None
        elif len(list_of_found_results) == 1:
            return list_of_found_results[0]
        else:
            raise ValueError(f"Multiple results with the same collection {collection} and retrieval model {retrieval_model} found in the reformulation object with name {self.reformulation_name}")
    
    def add_result(
            self, 
            collection:str, 
            retrieval_model:str, 
            result_dict: Dict) -> None:
        '''
        Adds a result to the reformulation.

        Args:
            result (Result): The result to add.
        '''
        result_found = self.find_result(collection, retrieval_model)
        if result_found is None:
            result = Result(collection, retrieval_model, result_dict)
            self.results.append(result)
        else:
            result_found.metrics = result_dict

    


class Topic:
    '''
    A class representing a topic.

    Attributes:
        id (int): The ID of the topic.
        domain (str): The domain of the topic.
        topic (str): The topic description.
        description (str): The description of the topic.
        narrative (str): The narrative of the topic.
        source (str): The source of the topic.
        concepts (List[str]): The list of concepts related to the topic.
        reformulations (List[Reformulation]): The list of reformulations for the topic.

    Methods:
        __init__: Initializes a new instance of the Topic class.
        __str__: Returns a string representation of the Topic object.
        __repr__: Returns a string representation of the Topic object.
        to_dict: Converts the Topic object to a dictionary.
        from_dict: Initializes the Topic object from a dictionary.
        find_reformulation: Finds a reformulation by its name.
        query_type_2_query: Generates/finds a query based on the specified query type.

    '''
    def __init__(self,
                    id: int = None,
                    domain: str = None,
                    topic: str = None,
                    description: str = None,
                    narrative: str = None,
                    source: str = None,
                    concepts: List[str] = None,
                    reformulations: List[Reformulation] = []
                    ) -> None:
        self.id = id
        self.domain = domain
        self.topic = topic
        self.description = description
        self.narrative = narrative
        self.source = source
        self.concepts = concepts
        self.reformulations = reformulations

    def __str__(self) -> str:
        print_str = f"id = {self.id}\n"
        print_str += f"Topic: {self.topic}\n"
        print_str += f"Description: {self.description}\n"
        print_str += f"Narrative: {self.narrative}\n"
        print_str += f"nb of reformulations: {len(self.reformulations)}\n"
        print_str += f"Source: {self.source}\n"

        return print_str
    
    def __repr__(self) -> str:
        return self.__str__()
    

    def to_dict(self) -> Dict:
        '''
        Converts the Topic object to a dictionary.

        Returns:
            dict: The dictionary representation of the Topic object.
        '''
        topic_dict = {}
        if self.id is not None:
            topic_dict["id"] = self.id
        if self.domain is not None:
            topic_dict["domain"] = self.domain
        if self.topic is not None:
            topic_dict["topic"] = self.topic
        if self.description is not None:
            topic_dict["description"] = self.description
        if self.narrative is not None:
            topic_dict["narrative"] = self.narrative
        if self.source is not None:
            topic_dict["source"] = self.source
        if self.concepts is not None:
            topic_dict["Concepts"] = self.concepts
        
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
                    if result.collection is not None:
                        result_dict["collection"] = result.collection   
                    reformulation_dict["results"].append(result_dict)
            reformulations.append(reformulation_dict)
        
        topic_dict["reformulations"] = reformulations

        return topic_dict


    def from_dict(self, topic_dict: Dict) -> None:
    
        '''
        Initializes the Topic object from a dictionary.

        Args:
            topic_dict (dict): The dictionary containing the topic information.
        '''
        if "id" in topic_dict:
            self.id = topic_dict["id"]
        if "domain" in topic_dict:
            self.domain = topic_dict["domain"]
        if "Concepts" in topic_dict:
            self.concepts = topic_dict["Concepts"]
        if "topic" in topic_dict:
            self.topic = topic_dict["topic"]
        if "description" in topic_dict:
            self.description = topic_dict["description"]
        if "narrative" in topic_dict:
            self.narrative = topic_dict["narrative"]
        if "source" in topic_dict:
            self.source = topic_dict["source"]
        if "Concepts" in topic_dict:
            self.concepts = topic_dict["Concepts"]
        
        # determine what are the reformulations
        reformulations = []

        # new format
        if "reformulations" in topic_dict:
            for reformulation in topic_dict["reformulations"]:
                reformulation_obj = Reformulation(
                    reformulation_name = reformulation["reformulation_name"],
                    reformulated_query = reformulation["reformulated_query"],
                    results=[]
                )

                for result_dict in reformulation["results"]:
                    reformulation_obj.add_result(
                        collection = result_dict["collection"],
                        retrieval_model = result_dict["retrieval_model"],
                        result_dict = result_dict["metrics"]
                    )
                reformulations.append(reformulation_obj)
        else:
            # old format
            for key in topic_dict.keys():
                if type(topic_dict[key]) == str:
                    if "reformulated" in key or "pseudo_narrative" in key:
                        reformulation = Reformulation()
                        reformulation.reformulated_query = topic_dict[key]
                        reformulation.reformulation_name = key
                        reformulation.results = []
                        reformulations.append(reformulation)


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
            raise ValueError(f"Multiple reformulations with the same name {reformulation_name} found in the topic object with id {self.id} and topic {self.topic}")
    

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


def save_topics_to_json(
        topics: List[Topic],
        output_topic_path: str
        ) -> List[Dict]:
    '''
    Save a list of Topic objects to a json file
    '''
    topic_list = [topic.to_dict() for topic in topics]
    
    with open(output_topic_path, 'w') as f:
        json.dump(topic_list, f, indent=4)
    
    return topic_list

        
def load_topics_from_json(
        input_topic_path: str, 
        range_start: int = 0,
        range_end: int = 150,
        ) -> List[Topic]:

    ### read json file
    with open(input_topic_path, 'r') as f:
        topics = json.load(f)
    
    ### resulting list of Topic objects
    topic_objects = []
    
    for topic in topics[range_start:range_end]:
        topic_object = Topic()
        topic_object.from_dict(topic) 
        topic_objects.append(topic_object)
    
    return topic_objects
    


                 
                
