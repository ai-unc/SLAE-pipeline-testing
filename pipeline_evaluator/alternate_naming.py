import json
import os
from openai import OpenAI

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

def rephrase(text) -> str:
  """Used to call an OpenAI LLM model"""
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "provide an alternate phrasing for the provided text. limit the response to the one alternate phrasing."},
        {"role": "user", "content": text}
    ]   
  )
  return completion.choices[0].message.content

for paper in os.listdir('pipeline_evaluator/full_dataset')[2:]:
    print(paper)
    with open(f'pipeline_evaluator/full_dataset/{paper}', 'r') as f:
        data = json.load(f)
        for relation in data["relations"]:
            relation["alternate_independent_variable_name"] = rephrase(relation["independent_variable_name"])
            relation["alternate_dependent_variable_name"] = rephrase(relation["dependent_variable_name"])
    with open(f'pipeline_evaluator/full_dataset/{paper}', 'w') as f:
        f.write(json.dumps(data, indent=2))
    
