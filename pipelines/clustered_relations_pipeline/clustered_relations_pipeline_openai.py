import json
import yaml
from openai import OpenAI
from langchain.pydantic_v1 import BaseModel, validator, Field
from typing import List,Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from sklearn.cluster import KMeans
from langchain.output_parsers import PydanticOutputParser
from random import shuffle
from statistics import mean, median, stdev
from os import listdir, getenv
from datetime import datetime

key = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

class SingleRelation(BaseModel):
    independent_variable_name: str
    dependent_variable_name: str
    is_causal: str
    attributes: str
    supporting_text: str
    relation_classification: str
    
    @validator("relation_classification")
    def allowed_classifications(cls, field):
        if field.lower() in {"direct", "inverse", "not applicable", "independent"}:
            return field
        else:
            raise ValueError(f"Invalid Relationship Type {{{field}}}")

class ListOfRelations(BaseModel):
  relations: list[SingleRelation]

class RelationCountError(Exception):
  def __init__(self, message):
    self.message = message
    super().__init__(self.message)

embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")

#Define additional functions for pipeline

def clean_data(data_path:str) -> dict:
    """Reads Json and removes list of user predictions"""
    with open(data_path, "r") as f:
        data = json.load(f)
    for relation in data['relations']:
        relation["relation_classification"] = ""
        relation["is_causal"] = ""
        relation["supporting_text"] = ""
    return data  

def extract_all_ordered_pairs(data:Dict) -> List[str]:
    #Extract the relationships
    relationships = data.get("relations", [])
    variable_pairs = []
    # Iterate through each relationship and extract variables
    for relationship in relationships:
        variable_one = relationship.get("independent_variable_name", "")
        variable_two = relationship.get("dependent_variable_name", "")
        variable_pairs.append(variable_one + " -> " + variable_two)
    return variable_pairs
  
def segment_text(text: str) -> List[str]:
    """
    Segments the given text into sections each containing one paragraph, separated by double newlines.
    """
    paragraphs = text.split('\n\n')
    print(f"PARAGRAPHS: {len(paragraphs)}")
    #remove empty segments
    paragraphs = [paragraph.replace("\n", " ") for paragraph in paragraphs if paragraph.strip() != "" and len(paragraph) >= 20]
    return paragraphs

def cluster_text(text:str, num_clusters:int) -> List[str]:
  """
  Splits text into a list of its paragraphs, then combines similar paragraphs into clusters and returns a list of clusters
  """
  print("Clustering text...")
  #split text into multiple sentence segments
  segments = segment_text(text)
  for segment in segments:
    print("--------------------------------")
    print(segment)
  # segments = [seg for seg in segments if seg.strip() != "" and len(seg) >= 10] #remove empty segments
  if len(segments) == 1: return segments  #return if only one paragraph
  
  segment_embeddings = embeddingModel.encode(segments)
  print("Embeddings complete.")
  # Cluster together similar segments using KMeans
  kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(segment_embeddings)
  print("KMeans clustering complete.")
  
  #Group segments into clusters based on KMean clustering
  clusters : Dict = {}
  for seg,k in zip(segments, kmeans.labels_):
    if k not in clusters: clusters[k] = ""
    clusters[k] += (seg) + "\n"
  for c in clusters:
    print(clusters[c])
  print("Clustering complete.")
  return list(clusters.values())

def call_LLM(instruction, text, model:str) -> str:
  """Used to call a OpenAI LLM model"""
  print("------------------------------------START PROMPT------------------------------------")
  print(text)
  print("-------------------------------------END PROMPT-------------------------------------")
  completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": text}
    ]   
  )
  return completion.choices[0].message.content

def summarize(text:str, model:str) -> str:
  """Used to summarize text during pre-processing
  Currently unused in Clustered Relations pipeline
  """
  prompt = \
    f"""
    You will be given exceprts of an academic paper. Your task is to summarize the main points of the paper.
    Include a comprehensive overview of the main research question, methodology, key findings, and conclusions.
    Emphasize the findings by detailing the data analysis methods used,
    significant results, how these results address the research question,
    and any implications or recommendations made by the authors.
    Also, mention any limitations of the study acknowledged by the authors.
    Conclude with the potential impact of this research in its respective field.
    Write your response in the form of one or multiple paragraphs.
    """
  return call_LLM(prompt, text, model)

def pipeline(data:Dict, instructions:str, text_format:str, model:str, *, debug:bool=False) -> Dict:
  """
  data should already be cleaned
  This pipeline uses embedding similarity to match relations to summaries
  """
  if debug: print(f"Model: {model}")
  paper_text = data["content"]
  
  # Extract relationships from the summarized text
  relationships:List[str] = extract_all_ordered_pairs(data)
  
  #Create clusters and summarize
  CLUSTER_MULTIPLIER:float = 1 #The number of clusters is (num relationships * this number).
  summaries = []
  clusters = cluster_text(paper_text, max(1, int(len(relationships)*CLUSTER_MULTIPLIER + 1))) # ensure at least one cluster
  print(len(clusters))
  
  summaries = [cluster for cluster in clusters]
    
  parser = PydanticOutputParser(pydantic_object=ListOfRelations)
  instructions = instructions.format(format_instructions=parser.get_format_instructions())
  
  summary_embeddings = embeddingModel.encode(summaries)
  
  # Batch process relationships
  BATCH_SIZE = 3
  SUMMARIES_PER_BATCH = 5
  output_jsons = []
  for i in range(0, len(relationships), BATCH_SIZE):
    relations = relationships[i:i+BATCH_SIZE]
    relation_embeddings = embeddingModel.encode("\n".join(relations))
    if debug:
      print(f"relation embedding shape: {relation_embeddings.shape}")
      print(f"summary embedding shape: {summary_embeddings.shape}")
    similarities = pytorch_cos_sim(relation_embeddings, summary_embeddings).squeeze()
    if debug:
      print("Similarities:")
      print(similarities.shape)
      print(similarities)
    indices = similarities.argsort(descending=True)[:SUMMARIES_PER_BATCH+1] 
    if debug:
      print("Indices:")
      print(indices)
    # summary = summarize("\n".join([summaries[i] for i in indices]), model)
    summary = "\n".join([summaries[i] for i in indices])
    input_text = text_format.format(text=summary, relationships="\n".join(relations), count=len(relations))
    output = call_LLM(instructions, input_text, model)
    output_jsons.append(output)
  
  if debug: print(f"Pipeline complete. Output:\n{output}")
  
  #ensure correct formatting and merge all outputs
  parsed_output = {"relations":[]}
  for output_json in output_jsons:
    parsed_json = parser.parse(output_json).dict()
    for relation in parsed_json["relations"]:
      if relation not in parsed_output["relations"]:
        parsed_output["relations"].append(relation)
  
  if debug: print(f"Parsed output:\n{parsed_output}")
  # ensure only desired relations are present

  final_output = {"relations":data["relations"]}
  #set all relations to a default of not applicable
  for relation in final_output["relations"]:
    relation["relation_classification"] = "not applicable"
  
  # Create a dictionary for quick lookup of relations
  parsed_relations_dict = {
    (relation["independent_variable_name"], relation["dependent_variable_name"]): relation
    for relation in parsed_output["relations"]
}

  # Update the final_output based on the dictionary
  for relation in final_output["relations"]:
    key = (relation["independent_variable_name"], relation["dependent_variable_name"])
    if key in parsed_relations_dict:
        parsed_relation = parsed_relations_dict[key]
        relation["relation_classification"] = parsed_relation["relation_classification"]
        relation["is_causal"] = parsed_relation["is_causal"]
        relation["supporting_text"] = parsed_relation["supporting_text"]
  
  return final_output

def call_pipeline(data_path:str, settings_path:str, *, use_alternate_names=False) -> Dict:
  with open(settings_path, "r") as f:
    pipeline_settings = yaml.safe_load(f)
    debug = pipeline_settings["verbose"]
    instructions = pipeline_settings["instructions"]
    text = pipeline_settings["prompt"]
    model = pipeline_settings["model"]
  data = clean_data(data_path)
  if use_alternate_names:
    for relation in data["relations"]:
      relation["independent_variable_name"] = relation["alternate_independent_variable_name"]
      relation["dependent_variable_name"] = relation["alternate_dependent_variable_name"]
  return pipeline(data, instructions, text, model, debug=debug) 


if __name__ == "__main__":
  EVALUATE = True
  RANDOMIZE = False
  DEBUG = True
  USE_ALTERNATE_NAMES = False
  NUM_TRIALS = 1
  NUM_PAPERS = 1
  
  def score(solution:List[Dict], submission:List[Dict]) -> List[float]:
    scores = {}
    for i, paper in enumerate(submission):
      ground_truth = solution[i]
      if DEBUG:
        print("\n\n\nGUESS:")
        print(*[r["relation_classification"] for r in paper["relations"]], sep="\n")
        print("\n\n\nGROUND TRUTH:")
        print(*[r["relation_classification"] for r in ground_truth["relations"]], sep="\n")
        
      if "relations" not in paper or "relations" not in ground_truth:
        raise Exception("Key error: relations.")
      if len(paper["relations"]) != len(ground_truth["relations"]):
        print("\n\n\nGUESS:")
        print(*extract_all_ordered_pairs(paper), sep="\n")
        print("\n\n\nGROUND TRUTH:")
        print(*extract_all_ordered_pairs(ground_truth), sep="\n")
        raise RelationCountError(f"Prediction has {len(paper['relations'])} relations and ground truth has {len(ground_truth['relations'])}.")
      
      score = 0
      for j, prediction in enumerate(paper["relations"]):
        relation = ground_truth["relations"][j]
        score += 1 if relation["relation_classification"].lower().strip() == prediction["relation_classification"].lower().strip() else 0
      paper_score = (score / len(ground_truth["relations"])) * 100
      scores[ground_truth["title"]] = paper_score
    
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
            print(f"Predicting {paper}")
            prediction = call_pipeline(data_path=f"./pipeline_evaluator/full_dataset/{paper}",
                                        settings_path="./pipelines/clustered_relations_pipeline/pipeline_settings_openai.yaml",
                                        use_alternate_names=USE_ALTERNATE_NAMES)
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