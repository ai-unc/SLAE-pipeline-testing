import pandas as pd
import json
import os
import random


def clean_data(data: dict, verbose=False) -> dict:
    """Reads Json and removes list of user predictions"""
    for relation in data['Relations']:
        relation["RelationshipClassification"] = ""
        relation["IsCausal"] = ""
        relation["SupportingText"] = ""
        relation["Attributes"] = ""
    return data  

def make_random_data(data: dict, verbose=False) -> dict:
    """Reads Json and removes list of user predictions"""
    for relation in data['Relations']:
        relation["RelationshipClassification"] = ["independent", "direct", "inverse", "not applicable"][random.randint(0, 3)]
        relation["IsCausal"] = ""
        relation["SupportingText"] = ""
        relation["Attributes"] = ""
    return data  

TEST_SET = False
RANDOMIZE = True

# Directory containing your JSON files
directory = "pipeline_evaluator/full_dataset"

# Read each JSON file as a string and store in a list
count = 0
json_strings = []
indexes = []
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        if count < 19:
            count += 1
            continue
        with open(os.path.join(directory, filename), 'r') as file:
            if TEST_SET:
                json_content = file.read()
                json_dict = json.loads(json_content)
                clean_json_dict = clean_data(json_dict)  # remove answers
                json_strings.append(json.dumps(clean_json_dict))
            elif RANDOMIZE:
                json_content = file.read()
                json_dict = json.loads(json_content)
                clean_json_dict = make_random_data(json_dict)  # remove answers
                json_strings.append(json.dumps(clean_json_dict))
            else:
                json_content = file.read()
                # Optional: Validate JSON content by loading it
                json.loads(json_content)
                json_strings.append(json_content)
            indexes.append(count)
        count += 1
# Create a DataFrame with one column containing all the JSON strings
if False:  # if making test solution
    combined_df = pd.DataFrame({'id':indexes, 'json_data':json_strings, 'Usage':["Public" if i/len(json_strings) > .3 else "Private" for i in range(len(json_strings))]})
else:
    combined_df = pd.DataFrame({'id':indexes, 'json_data':json_strings})

# combined_df.set_index('id', inplace=True)

# Save the DataFrame to a CSV file, choosing a delimiter unlikely to appear in your data
combined_df.to_csv("sample_submission.csv", index=False)