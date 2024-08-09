import google.generativeai as genai
import json
import yaml
from langchain.pydantic_v1 import BaseModel, validator
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from google.api_core.exceptions import ResourceExhausted
from time import sleep
from os import getenv
from datetime import datetime

key = getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)

class SingleRelation(BaseModel):
    VariableOneName: str
    VariableTwoName: str
    SupportingText: str
    isCausal: str
    RelationshipClassification: str
    
    @validator("RelationshipClassification")
    def allowed_classifications(cls, field):
        if field.lower() in {"direct", "inverse", "not applicable", "independent"}:
            return field
        else:
            raise ValueError(f"Invalid Relationship Type {{{field}}}")

class ListOfRelations(BaseModel):
    Relations: list[SingleRelation]

class RelationCountError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# Model parameters
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 16000,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

def clean_data(data_path:str) -> dict:
    """Reads Json and removes list of user predictions"""
    with open(data_path, "r") as f:
        data = json.load(f)
    for relation in data['Relations']:
        relation["RelationshipClassification"] = ""
        relation["isCausal"] = ""
        relation["SupportingText"] = ""
    return data  

def extract_single_relation(data:Dict) -> Dict[str, str]:
    # Extract the first relationship only
    relationships = data.get("Relations", [])
    if relationships:
        relationship = relationships[0]
        return {
            "VariableOneName": relationship.get("VariableOneName", ""),
            "VariableTwoName": relationship.get("VariableTwoName", "")
        }
    return {}

def call_LLM(text, model:genai.GenerativeModel) -> str:
    """Used to call a Google LLM model with ResourceExhausted handling"""
    try:
        result = model.generate_content(text)
        return result.text
    except ResourceExhausted:
        print("ResourceExhausted error. Retrying in 15 seconds...")
        sleep(15)
        return call_LLM(text, model)
    except ValueError as e:
        print(result)
        raise ValueError(f"ValueError: {e}")

def summarize(text:str, model:genai.GenerativeModel) -> str:
    """Used to summarize text during pre-processing"""
    prompt = \
        f"""
        {text}

        Please provide a detailed summary of the above academic paper.
        Include a comprehensive overview of the main research question, methodology, key findings, and conclusions.
        Emphasize the findings by detailing the data analysis methods used,
        significant results, how these results address the research question,
        and any implications or recommendations made by the authors.
        Also, mention any limitations of the study acknowledged by the authors.
        Conclude with the potential impact of this research in its respective field.
        Write your response in the form of multiple paragraphs.
        """
    return call_LLM(prompt, model)

def pipeline(data:Dict, model:genai.GenerativeModel, prompt:str, *, debug:bool=False) -> Dict:
    """
    data should already be cleaned
    """
    paper_text = data["PaperContents"]

    # Remove references (optional)
    paper_text = paper_text.rpartition("References")[0]
    if debug: print(f"Paper text:\n{paper_text}")

    # Summarize paper
    if debug: print("Summarizing paper...")
    paper_text = summarize(paper_text, model)  
    if debug: print(f"Summarized text:\n{paper_text}")

    # Extract single relationship
    single_relation = extract_single_relation(data)
    if not single_relation:
        raise ValueError("No relationships found in data.")
    
    parser = PydanticOutputParser(pydantic_object=SingleRelation) # Refers to a class called SingleRelation.
    prompt_template = PromptTemplate(
                          template=prompt,
                          input_variables=["text", "relationship"],
                          partial_variables={"format_instructions": parser.get_format_instructions}
                          )
    
    input_text = prompt_template.format_prompt(
        text=paper_text, 
        relationship=f"{single_relation['VariableOneName']} -> {single_relation['VariableTwoName']}"
    ).to_string()

    if debug: print(f"Input prompt:\n{input_text}")
    batch_output = call_LLM(input_text, model)
    parsed_output = parser.parse(batch_output)
    if debug: print(f"Output:\n{parsed_output.dict()}")

    # Integrate the result into the original data
    final_output = {"Relations": [single_relation]}
    final_output["Relations"][0]["RelationshipClassification"] = parsed_output.RelationshipClassification
    final_output["Relations"][0]["isCausal"] = parsed_output.isCausal
    final_output["Relations"][0]["SupportingText"] = parsed_output.SupportingText

    return final_output

def call_pipeline(data_path, settings_path:str) -> Dict:
    with open(settings_path, "r") as f:
        pipeline_settings = yaml.safe_load(f)
        debug = pipeline_settings["verbose"]
        prompt = pipeline_settings["prompt"]
        model = genai.GenerativeModel(model_name=pipeline_settings["model"],
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
    data = clean_data(data_path)
    return pipeline(data, model, prompt, debug=debug)

if __name__ == "__main__":
    EVALUATE = True
    DEBUG = True

    if EVALUATE:
        data_path = "./pipeline_evaluator/full_dataset/example_paper.json"  # Use a single paper for testing
        settings_path = "./pipelines/iterative_summary_pipeline/pipeline_settings.yaml"
        
        prediction = call_pipeline(data_path=data_path, settings_path=settings_path)
        
        if DEBUG:
            print("\nFinal Prediction:")
            print(json.dumps(prediction, indent=2))
