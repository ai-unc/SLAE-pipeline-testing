import json
import os


def main(paper_testfile_path):
    file = open(paper_testfile_path)
    data = json.load(file)
    content_lowercase = data["content"].lower()
    index = content_lowercase.rfind("\nresults\n")
    if index == -1:
        index = content_lowercase.rfind("\nresults:\n")
    if index == -1:
        index = content_lowercase.rfind("\nresults:")
    if index == -1:
        index = content_lowercase.rfind("results\n\n")

    if index == -1:
        index = content_lowercase.rfind("\nconclusions:\n")
    if index == -1:
        index = content_lowercase.rfind("\nconclusions:\n")
    if index == -1:
        index = content_lowercase.rfind("\nconclusions:")
    if index == -1:
        index = content_lowercase.rfind("conclusions\n\n")

    if index == -1:
        index = content_lowercase.rfind("\nconclusion:\n")
    if index == -1:
        index = content_lowercase.rfind("\nconclusion:\n")
    if index == -1:
        index = content_lowercase.rfind("\nconclusion:")
    if index == -1:
        index = content_lowercase.rfind("conclusion\n\n")

    if index == -1:
        print(data["title"])
    return


if __name__ == "__main__":
    cwd = "./pipeline_evaluator/full_dataset"
    for file_name in os.listdir(cwd):
        try:
            main(os.path.join(cwd, file_name))
        except:
            print("failed on", os.path.join(cwd, file_name))
