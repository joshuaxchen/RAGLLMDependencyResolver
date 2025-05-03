import os
import json
import re

def get_metrics():
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, 'repository_analysis_results.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total_correct = 0
    recall_missed = 0
    precision_missed = 0
    for repo in results:
        repo_packages = set()
        repo_name = os.path.basename(repo['repository'])
        print(f"Repository: {repo_name}")
        data = extract_json(repo['packages'])
        if data == None: continue
        packages = data.get("python_packages", [])
        for pkg in packages:
            name = pkg['package']
            version = pkg['version']
            repo_packages.add(name)
        deps = get_correct_packages(repo_name)
        print(repo_packages)
        print(deps)
        correct = 0
        for name in deps:
            m = re.match(r'^([A-Za-z0-9_.\-]+)', name)
            if m != None and any(pkg in name for pkg in repo_packages):
                print(m.group(1))
                correct += 1
        recall_missed += len(deps) - correct
        precision_missed += len(repo_packages) - correct
        total_correct += correct
    precision = total_correct / (precision_missed + total_correct)
    recall = total_correct / (recall_missed + total_correct)
    print(f"\nTotal correct: {total_correct} recall missed: {recall_missed}")
    print(f"\nPrecision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {2 * precision * recall / (precision + recall)}")
        


def get_correct_packages(repo_name):
    deps = []
    base_dir = os.path.dirname(__file__)
    json_pathl = os.path.join(base_dir, 'dataset-dibench-regular.jsonl')
    with open(json_pathl, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if obj['instance_id'] == repo_name:
                patch = obj['patch'].splitlines()
                
                in_block = False
                for l in patch:
                    if l.startswith("+dependencies"):
                        in_block = True
                        continue

                    if in_block:
                        if l.startswith("+]"):
                            break
                        if l.startswith("+"):
                            # strip the "+", whitespace and trailing comma, then strip quotes
                            item = l.lstrip("+ ").rstrip(",").strip().strip('"')
                            deps.append(item)
    return deps


def extract_json(raw: str) -> str:
    try:
        start = raw.find('{')
        end   = raw.rfind('}')
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object braces found in input")

        json_str = raw[start : end+1]
        return json.loads(json_str)
    except Exception as e:
        return None


if __name__ == "__main__":
    get_metrics()