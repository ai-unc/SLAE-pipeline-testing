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

def cluster_text(text:str) -> List[str]:
  """
  Splits text into a list of its paragraphs, then combines similar paragraphs into clusters and returns a list of clusters
  """
  segments : List[str] = text.split(".\n")    #split text into paragraphs
  if len(segments) == 1: return segments  #return if only one paragraph
  
  segment_embeddings = embeddingModel.encode(segments)
    
  # Cluster together similar segments using KMeans
  CLUSTER_DIVISOR:int = 3  # <- this is a hyperparameter that could be tuned
  num_clusters = max(1, len(segments) // CLUSTER_DIVISOR) # make sure there is at least one cluster
  kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(segment_embeddings)

  #Group segments into clusters based on KMean clustering
  clusters : Dict = {}
  for seg,k in zip(segments, kmeans.labels_):
    if k not in clusters: clusters[k] = ""
    clusters[k] += (seg) + "\n"
  return list(clusters.values())

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
  prompt = "Please provide a detailed summary of the above text in the form of a paragraph." # <- prompt engineering would probably help here
  return call_LLM(text + "\n\n\n" + prompt, model)

def pipeline_old(data:Dict, model:genai.GenerativeModel, prompt:str, *, num_iterations:int, verbose:bool=False) -> Dict:
  """
  data should already be cleaned
  """
  paper_text = data["PaperContents"]
  data["PaperContents"] = ""  # Remove paper fulltext from output to avoid clogging telemetry
  
  # Summarize paper using specified iterations
  for i in range(num_iterations):
    if verbose: print(f"clustering for iteration {i+1}")
    clusters = cluster_text(paper_text)
    paper_text = ""
    if verbose: print(f"summarizing for iteration {i+1}")
    for cluster in clusters:
      summary = summarize(cluster, model)
      paper_text += summary.text
    if verbose: print(f"Iteration {i+1} of {num_iterations} complete")
    
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

def pipeline(data:Dict, model:genai.GenerativeModel, prompt:str, *, num_iterations:int, verbose:bool=False) -> Dict:
  """
  data should already be cleaned
  This pipeline uses embedding similarity to match relations to summaries
  """
  paper_text = data["PaperContents"]
  data["PaperContents"] = ""  # Remove paper fulltext from output to avoid clogging telemetry
  
  # Summarize paper using specified iterations
  # NOTE: this method keeps all layers of summaries
  summaries = []
  for i in range(num_iterations):
    if verbose: print(f"clustering for iteration {i+1}")
    clusters = cluster_text(paper_text)
    paper_text = ""
    if verbose: print(f"summarizing for iteration {i+1}")
    for cluster in clusters:
      summary = summarize(cluster, model)
      summaries.append(summary.text)
      paper_text += summary.text
    if verbose: print(f"Iteration {i+1} of {num_iterations} complete")
    
  # Extract relationships from the summarized text
  relationships:List[str] = extract_all_ordered_pairs(data)
  parser = PydanticOutputParser(pydantic_object=ListOfRelations) #Refers to a class called SingleRelation.
  prompt_template = PromptTemplate(
                          template=prompt,
                          input_variables=["text", "relationships"],
                          partial_variables={"format_instructions":parser.get_format_instructions}
                          )
  
  summary_embeddings = embeddingModel.encode(summaries)
  relation_embeddings = embeddingModel.encode(relationships)
  
  # Allocate relations to their most relevant summary
  allocated_relations = {s:[] for s in summaries}
  for i,relation in enumerate(relationships):
    similarities = pytorch_cos_sim(relation_embeddings[i], summary_embeddings)
    most_similar = summaries[similarities.argmax()]
    allocated_relations[most_similar].append(relation)
  
  output_jsons = []
  for summary,relations in allocated_relations.items():
    if(len(relations) == 0): continue
    input_text = prompt_template.format_prompt(text=summary, relationships="\n".join(relations)).to_string()
    output = call_LLM(input_text, model)
    output_jsons.append(output)
    if verbose: print(f"Summary complete. Output:\n{output.text}")
  
  if verbose: print(f"Pipeline complete. Output:\n{output.text}")
  
  #ensure correct formatting and merge all outputs
  parsed_output = {"Relations":[]}
  for oj in output_jsons:
    parsed_json = parser.parse(oj.text).dict()
    for relation in parsed_json["Relations"]:
      if relation not in parsed_output["Relations"]:
        parsed_output["Relations"].append(relation)
  
  return parsed_output


def call_pipeline(data_path:str, settings_path:str) -> Dict:
  with open(settings_path, "r") as f:
    pipeline_settings = yaml.safe_load(f)
    verbose = pipeline_settings["verbose"]
    prompt = pipeline_settings["prompt"]
    model = genai.GenerativeModel(model_name=pipeline_settings["model"],
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
  data = clean_data(data_path)
  return pipeline(data, model, prompt, num_iterations=1, verbose=verbose) # Toggle here between pipeline_old and pipeline for different methods


if __name__ == "__main__":
  EVALUATE = True
  RANDOMIZE = True
  DEBUG = True
  NUM_TRIALS = 3
  NUM_PAPERS = 3
  
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