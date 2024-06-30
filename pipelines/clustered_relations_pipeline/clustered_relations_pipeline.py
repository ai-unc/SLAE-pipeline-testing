import google.generativeai as genai
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
from os import listdir, getenv
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

def cluster_text(text:str, num_clusters:int) -> List[str]:
  """
  Splits text into a list of its paragraphs, then combines similar paragraphs into clusters and returns a list of clusters
  """
  segments : List[str] = text.split(".\n")    #split text into paragraphs
  if len(segments) == 1: return segments  #return if only one paragraph
  
  segment_embeddings = embeddingModel.encode(segments)
    
  # Cluster together similar segments using KMeans
  kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(segment_embeddings)

  #Group segments into clusters based on KMean clustering
  clusters : Dict = {}
  for seg,k in zip(segments, kmeans.labels_):
    if k not in clusters: clusters[k] = ""
    clusters[k] += (seg) + "\n"
  return list(clusters.values())

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
    
    Please provide a detailed summary of the above academic paper excerpts.
    Include a comprehensive overview of the main research question, methodology, key findings, and conclusions.
    Emphasize the findings by detailing the data analysis methods used,
    significant results, how these results address the research question,
    and any implications or recommendations made by the authors.
    Also, mention any limitations of the study acknowledged by the authors.
    Conclude with the potential impact of this research in its respective field.
    Write your response in the form of one or multiple paragraphs.
    """
  return call_LLM(prompt, model)

def pipeline(data:Dict, model:genai.GenerativeModel, prompt:str, *, debug:bool=False) -> Dict:
  """
  data should already be cleaned
  This pipeline uses embedding similarity to match relations to summaries
  """
  if debug: print(f"Model: {model.model_name}")
  paper_text = data["PaperContents"]
  # remove references
  # should probably be redone with a more robust method, maybe regex
  paper_text = paper_text.rpartition("References")[0]
  
  # Extract relationships from the summarized text
  relationships:List[str] = extract_all_ordered_pairs(data)
  
  #Create clusters and summarize
  CLUSTER_MULTIPLIER:float = 3 #The number of clusters is (num relationships * this number).
  summaries = []
  clusters = cluster_text(paper_text, max(1, int(len(relationships)*CLUSTER_MULTIPLIER + 1))) # ensure at least one cluster
  paper_text = ""
  for cluster in clusters:
    summary = summarize(cluster, model)
    summaries.append(summary)
    
  parser = PydanticOutputParser(pydantic_object=ListOfRelations)
  prompt_template = PromptTemplate(
                          template=prompt,
                          input_variables=["text", "relationships", "count"],
                          partial_variables={"format_instructions":parser.get_format_instructions}
                          )
  
  summary_embeddings = embeddingModel.encode(summaries)
  """OLD METHOD
  
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
    input_text = prompt_template.format_prompt(text=summary, relationships="\n".join(relations), count=len(relations)).to_string()
    output = call_LLM(input_text, model)
    output_jsons.append(output)
    if debug: print(f"Summary complete. Output:\n{output}")
  """
  BATCH_SIZE = 3
  SUMMARIES_PER_BATCH = 10
  output_jsons = []
  for i in range(0, len(relationships), BATCH_SIZE):
    relations = relationships[i:i+BATCH_SIZE]
    relation_embeddings = embeddingModel.encode(relations)
    similarities = pytorch_cos_sim(relation_embeddings, summary_embeddings)
    indices = similarities.argsort()[-SUMMARIES_PER_BATCH:] # sorted ascending accuracy
    input_text = prompt_template.format_prompt(text="\n".join([summaries[i] for i in indices]), relationships="\n".join(relations), count=len(relations)).to_string()
    output = call_LLM(input_text, model)
    output_jsons.append(output)
  
  if debug: print(f"Pipeline complete. Output:\n{output}")
  
  #ensure correct formatting and merge all outputs
  parsed_output = {"Relations":[]}
  for output_json in output_jsons:
    parsed_json = parser.parse(output_json).dict()
    for relation in parsed_json["Relations"]:
      if relation not in parsed_output["Relations"]:
        parsed_output["Relations"].append(relation)
  
  if debug: print(f"Parsed output:\n{parsed_output}")
  # ensure only desired relations are present

  final_output = {"Relations":data["Relations"]}
  #set all relations to a default of not applicable
  for relation in final_output["Relations"]:
    relation["RelationshipClassification"] = "not applicable"
  
  # Create a dictionary for quick lookup of relations
  parsed_relations_dict = {
    (relation["VariableOneName"], relation["VariableTwoName"]): relation
    for relation in parsed_output["Relations"]
}

  # Update the final_output based on the dictionary
  for relation in final_output["Relations"]:
    key = (relation["VariableOneName"], relation["VariableTwoName"])
    if key in parsed_relations_dict:
        parsed_relation = parsed_relations_dict[key]
        relation["RelationshipClassification"] = parsed_relation["RelationshipClassification"]
        relation["isCausal"] = parsed_relation["isCausal"]
        relation["SupportingText"] = parsed_relation["SupportingText"]
  
  return final_output

def call_pipeline(data_path:str, settings_path:str) -> Dict:
  with open(settings_path, "r") as f:
    pipeline_settings = yaml.safe_load(f)
    debug = pipeline_settings["verbose"]
    prompt = pipeline_settings["prompt"]
    model = genai.GenerativeModel(model_name=pipeline_settings["model"],
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
  data = clean_data(data_path)
  return pipeline(data, model, prompt, debug=debug) # Toggle here between pipeline_old and pipeline for different methods


if __name__ == "__main__":
  EVALUATE = True
  RANDOMIZE = False
  DEBUG = True
  NUM_TRIALS = 1
  NUM_PAPERS = 5
  
  def score(solution:List[Dict], submission:List[Dict]) -> List[float]:
    scores = {}
    for i, paper in enumerate(submission):
      ground_truth = solution[i]
      if DEBUG:
        print("\n\n\nGUESS:")
        print(*[r["RelationshipClassification"] for r in paper["Relations"]], sep="\n")
        print("\n\n\nGROUND TRUTH:")
        print(*[r["RelationshipClassification"] for r in ground_truth["Relations"]], sep="\n")
        
      if "Relations" not in paper or "Relations" not in ground_truth:
        raise Exception("Key error: Relations.")
      if len(paper["Relations"]) != len(ground_truth["Relations"]):
        print("\n\n\nGUESS:")
        print(*extract_all_ordered_pairs(paper), sep="\n")
        print("\n\n\nGROUND TRUTH:")
        print(*extract_all_ordered_pairs(ground_truth), sep="\n")
        raise RelationCountError(f"Prediction has {len(paper['Relations'])} relations and ground truth has {len(ground_truth['Relations'])}.")
      
      score = 0
      for j, prediction in enumerate(paper["Relations"]):
        relation = ground_truth["Relations"][j]
        score += 1 if relation["RelationshipClassification"].lower().strip() == prediction["RelationshipClassification"].lower().strip() else 0
      paper_score = (score / len(ground_truth["Relations"])) * 100
      scores[ground_truth["PaperTitle"]] = paper_score
    
    return scores
  
  if EVALUATE:
    trial_scores = []
      
    for _ in range(NUM_TRIALS):
        # Load data
        source = [file for file in listdir("./pipeline_evaluator/full_dataset")]
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
        eval_scores = score(papers, predictions)
        trial_scores.append(mean(eval_scores.values()))
    print("\n\n\n")
    if len(trial_scores) == 1:
      for title, score_ in eval_scores.items():
        print(f"{title}: {score_}")
      if len(eval_scores) > 1:
        print(f"Average accuracy score: {mean(eval_scores.values())}")
        print(f"Median accuracy score: {median(eval_scores.values())}")
        print(f"Standard deviation: {stdev(eval_scores.values())}")
    else:
      print("Number of trials:", NUM_TRIALS)
      print(f"Accuracy scores: {trial_scores}")
      print(f"Average accuracy score: {mean(trial_scores)}")
      print(f"Median accuracy score: {median(trial_scores)}")
      print(f"Standard deviation: {stdev(trial_scores)}")
    with open(f"./pipeline_evaluator/results/results.txt", "a") as f:
      f.write(f"{__file__.rpartition('/')[-1]}-{datetime.now().strftime('%m/%d/%Y-%H:%M:%S')} : {eval_scores}\n")