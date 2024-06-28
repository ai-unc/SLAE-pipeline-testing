import google.generativeai as genai
import os
import pandas as pd
import json
import yaml
from langchain.pydantic_v1 import BaseModel, validator, Field
from typing import List,Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from sklearn.cluster import KMeans
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from google.api_core.exceptions import ResourceExhausted
from time import sleep
from random import shuffle
from statistics import mean, median, stdev

key = os.getenv("GOOGLE_API_KEY")
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

embeddingModel = SentenceTransformer("all-mpnet-base-v2")

#Define additional functions for pipeline

def clean_data(data_path:str) -> dict:
    """Reads Json and removes list of user predictions"""
    with open(data_path, "r") as f:
        data = json.load(f)
    for relation in data['Relations']:
        relation["RelationshipClassification"] = ""
        relation["isCausal"] = ""
        relation["SupportingText"] = ""
    return data  

def extract_all_ordered_pairs(data:Dict) -> List[str]:
    #Extract the relationships
    relationships = data.get("Relations", [])
    variable_pairs = []
    # Iterate through each relationship and extract variables
    for relationship in relationships:
        variable_one = relationship.get("VariableOneName", "")
        variable_two = relationship.get("VariableTwoName", "")
        variable_pairs.append(variable_one + " -> " + variable_two)
    return variable_pairs

def call_LLM(text, model:genai.GenerativeModel) -> str:
  """Used to call the LLM model with ResourceExhausted handling"""
  try:
    result = model.generate_content(text)
    return result
  except ResourceExhausted:
    print("ResourceExhausted error. Retrying in 15 seconds...")
    sleep(15)
    return call_LLM(text, model)
    
def summarize(text:str, model:genai.GenerativeModel) -> str:
  """Used to summarize text during pre-processing"""
  prompt = \
    """
    {text}
    
    Please provide a detailed summary of the above academic paper.
    Include a comprehensive overview of the main research question, methodology, key findings, and conclusions.
    Emphasize the findings by detailing the data analysis methods used,
    significant results, how these results address the research question,
    and any implications or recommendations made by the authors.
    Also, mention any limitations of the study acknowledged by the authors.
    Conclude with the potential impact of this research in its respective field.
    """
  return call_LLM(prompt, model)

def pipeline(data:Dict, model:genai.GenerativeModel, prompt:str, *, verbose:bool=False) -> Dict:
  """
  data should already be cleaned
  """
  paper_text = data["PaperContents"]
  data["PaperContents"] = ""  # Remove paper fulltext from output to avoid clogging telemetry
  
  # Summarize paper
  if verbose: print("Summarizing paper...")
  
  paper_text = summarize(paper_text, model)  
    
  # Extract relationships from the summarized text
  relationships:str = "Below is a list of relationships:\n"
  relationships += "\n".join(extract_all_ordered_pairs(data))
  parser = PydanticOutputParser(pydantic_object=ListOfRelations) #Refers to a class called SingleRelation.
  prompt_template = PromptTemplate(
                          template=prompt,
                          input_variables=["text", "relationships"],
                          partial_variables={"format_instructions":parser.get_format_instructions}
                          )

  input_text = prompt_template.format_prompt(text=paper_text, relationships=relationships).to_string()
  output = call_LLM(input_text, model)
  
  if verbose: print(f"Pipeline complete. Output:\n{output.text}")
  
  # Ensure content is in valid json format with parser.
  parsed_output = parser.parse(output.text)
  return parsed_output.dict()

def call_pipeline(data_path, settings_path:str) -> Dict:
  with open(settings_path, "r") as f:
    pipeline_settings = yaml.safe_load(f)
    verbose = pipeline_settings["verbose"]
    prompt = pipeline_settings["prompt"]
    model = genai.GenerativeModel(model_name=pipeline_settings["model"],
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
  data = clean_data(data_path)
  return pipeline(data, model, prompt, verbose=verbose) # Toggle here between pipeline_old and pipeline for different methods

if __name__ == "__main__":
  EVALUATE = True
  RANDOMIZE = True
  DEBUG = True
  NUM_TRIALS = 1
  NUM_PAPERS = 10
  
  def score(solution:List[Dict], submission:List[Dict]) -> float:
    total_score = 0
    total_relations = 0
    for i, paper in enumerate(submission):
      ground_truth = solution[i]
      if DEBUG:
        print("\n\n\nGUESS:")
        print(*[r["RelationshipClassification"] for r in paper["Relations"]], sep="\n")
        print("\n\n\nGROUND TRUTH:")
        print(*[r["RelationshipClassification"] for r in ground_truth["Relations"]], sep="\n")
        
      if "Relations" not in paper or "Relations" not in ground_truth:
        raise Exception("Key error: Relations. Check with organizers.")
      if len(paper["Relations"]) != len(ground_truth["Relations"]):
        print("\n\n\nGUESS:")
        print(*extract_all_ordered_pairs(paper), sep="\n")
        print("\n\n\nGROUND TRUTH:")
        print(*extract_all_ordered_pairs(ground_truth), sep="\n")
        raise Exception(f"Prediction has {len(paper['Relations'])} relations and ground truth has {len(ground_truth['Relations'])}.")
      
      total_relations += len(ground_truth["Relations"])
        
      for j, prediction in enumerate(paper["Relations"]):
        relation = ground_truth["Relations"][j]
        total_score += 1 if relation["RelationshipClassification"].lower().strip() == prediction["RelationshipClassification"].lower().strip() else 0

    final_score = total_score / total_relations
    return final_score * 100
  
  if EVALUATE:
    accuracy_scores = []
      
    for _ in range(NUM_TRIALS):
        # Load data
        source = [file for file in os.listdir("./pipeline_evaluator/full_dataset")]
        if RANDOMIZE:
          shuffle(source)
        
        # Make Predictions
        predictions = []
        for paper in source[:NUM_PAPERS]:
            prediction = call_pipeline(data_path=f"./pipeline_evaluator/full_dataset/{paper}",
                                        settings_path="./pipelines/iterative_summary_pipeline/pipeline_settings.yaml")
            
            predictions.append(prediction)
        
        #score
        papers = []
        for p in source[:NUM_PAPERS]:
            with open(f"./pipeline_evaluator/full_dataset/{p}") as f:
                papers.append(json.load(f))
        eval_score = score(papers, predictions)
        accuracy_scores.append(eval_score)
    print("\n\n\n")
    print("Number of trials:", NUM_TRIALS)
    print(f"Accuracy scores: {accuracy_scores}")
    print(f"Average accuracy score: {mean(accuracy_scores)}")
    print(f"Median accuracy score: {median(accuracy_scores)}")
    print(f"Standard deviation: {stdev(accuracy_scores)}")
        
        
        