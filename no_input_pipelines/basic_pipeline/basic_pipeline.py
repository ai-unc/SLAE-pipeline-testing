import yaml
import json
from openai import OpenAI
from langchain.pydantic_v1 import BaseModel, validator, Field
from os import getenv, listdir
from langchain.output_parsers import PydanticOutputParser

"""
This pipeline is designed to generate a list of relationships present in a given text. 
Due to the nature of the pipeline, it is not possible to verify the accuracy of its output,
and it should only be used as a starting point.
"""

class SingleRelation(BaseModel):
    independent_variable_name: str
    dependent_variable_name: str
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

key = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

parser = PydanticOutputParser(pydantic_object=ListOfRelations)

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

def pipeline(paper:str):
  """Pipeline to generate a list of relationships present in a given text"""
  with open("no_input_pipelines/basic_pipeline/pipeline_settings.yaml", "r") as f:
    pipeline_settings = yaml.safe_load(f)
  instructions = pipeline_settings["instructions"].format(format_instructions=parser.get_format_instructions())
  # text is assigned this way to allow the possibility to add something before/after text
  text = pipeline_settings["prompt"].format(text=paper)
  model = pipeline_settings["model"]
  response = call_LLM(instructions, text, model)
  return parser.parse(response).dict()

if __name__ == "__main__":
  NUM_PAPERS = 1
  papers = listdir("pipeline_evaluator/full_dataset")
  for paper in papers[:NUM_PAPERS]:
    with open(f"pipeline_evaluator/full_dataset/{paper}", "r") as f:
      paper_text = json.load(f)["content"]
    print(pipeline(paper_text))