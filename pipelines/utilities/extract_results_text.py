import json
import os


def build_search_strings(base_search_string: str):
    search_strings = ["\n" + base_search_string + "\n"]
    search_strings += [base_search_string + "\n\n"]
    # + :
    search_strings += ["\n" + base_search_string + ":" + "\n"]
    search_strings += [base_search_string + ":" + "\n\n"]
    search_strings += [base_search_string + ":" + "\n"]
    search_strings += [base_search_string + ":"]
    return search_strings


def extract_results_text(paper_testfile_path):
    print("\033[95mProcessing paper")
    file = open(paper_testfile_path)
    data = json.load(file)
    content_lowercase = data["content"].lower()

    index = -1
    index_search_strings = (
        build_search_strings("results")
        + build_search_strings("conclusions")
        + build_search_strings("conclusion")
    )
    for search_string in index_search_strings:
        temp = content_lowercase.rfind(search_string)
        if temp < index: continue
        index = temp
        # If we find the search string beyond 40% of the text, ideally that should be where we can start looking for talk of the results. That seemed to be the case with manual testing
        # This also means that if a paper has a "Results" heading, it won't be overridden by looping through and also matching an even later "Conclusions" heading
        if index > len(content_lowercase) * 2 / 5:
            break
        elif index > -1:
            print(
                f"\033[93mSearch string found too early in content: {repr(search_string)} {index} / {len(content_lowercase)} \033[0;0m"
            )

    if index == -1:
        print(f"\033[91m{data["title"]}\033[0;0m: first index not found")
        print("\n")
        return

    last_index = len(content_lowercase)
    last_index_search_strings = (
        build_search_strings("references")
        + build_search_strings("acknowledgements")
        + build_search_strings("acknowledgement")
        + build_search_strings("contributions")
        + build_search_strings("works cited")
        + build_search_strings("literature cited")
    )
    for search_string in last_index_search_strings:
        temp = content_lowercase[index:].rfind(search_string)
        if temp > -1 and temp < last_index and temp > 400:
            last_index = temp

    if last_index < len(content_lowercase):
        last_index += index

    output = data["content"][index:last_index]
    if len(output) < 1:
        print(f"\033[91m{data["title"]}\033[0;0m: output empty")
    else:
        print(f"\033[32m{data["title"]}\033[0;0m")
        print(f"Index: {index}, Last Index: {last_index}")
        print(f"Output Length: {len(output)}")
    print("\n")
    return


if __name__ == "__main__":
    cwd = "./pipeline_evaluator/full_dataset"
    for file_name in os.listdir(cwd):
        try:
            extract_results_text(os.path.join(cwd, file_name))
        except:
            print("failed on", os.path.join(cwd, file_name))
