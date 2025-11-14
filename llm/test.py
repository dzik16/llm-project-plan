import json
import requests
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

# =============================
# 🔧 Fungsi: Check Ollama & Model
# =============================


def check_ollama_model(model_name: str = "llama3.2:3b") -> bool:
    """Check apakah Ollama running dan model tersedia"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama tidak running")
            print("💡 Jalankan: ollama serve")
            return False

        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]

        if model_name not in model_names:
            print(f"❌ Model '{model_name}' tidak terinstall")
            print(f"💡 Jalankan: ollama pull {model_name}")
            return False

        print(f"✅ Ollama running, model '{model_name}' tersedia")
        return True

    except requests.exceptions.ConnectionError:
        print("❌ Tidak dapat terhubung ke Ollama")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


# =============================
# 🧠 Fungsi: Ekstraksi dengan LLM (Fixed)
# =============================
def extract_with_llm(json_data: Dict, model: str = "llama3.2:3b"):
    """Ekstraksi menggunakan LLM dengan explicit task-to-resource mapping"""

    project_name = json_data.get("project", {}).get("name", "Unknown Project")
    tasks = json_data.get("tasks", [])

    # Build explicit mapping per person
    resource_mapping = defaultdict(list)
    for task in tasks:
        resource = task.get("resource", "").strip()
        if resource:
            resource_mapping[resource].append({
                "id": task.get("id"),
                "task_name": task.get("task_name"),
                "start_date": task.get("start_date"),
                "finish_date": task.get("finish_date"),
            })

    # Create very explicit prompt with per-person breakdown
    person_sections = []
    for resource, task_list in resource_mapping.items():
        task_details = "\n".join([
            f"   - Task: {t['task_name']} | Start: {t['start_date']} | Finish: {t['finish_date']}"
            for t in task_list
        ])
        person_sections.append(f"""
Person: {resource}
Number of tasks: {len(task_list)}
Tasks:
{task_details}
""")

    people_breakdown = "\n".join(person_sections)

    prompt = f"""You are a data extraction assistant. Extract work information for each person.

Project: {project_name}

{people_breakdown}

STRICT RULES:
1. Create ONE entry per person
2. Convert task names to LOWERCASE
3. Include ALL start dates and finish dates for each task in arrays
4. Calculate duration_days: (latest finish_date - earliest start_date) + 1
5. Calculate kompleksitas based on total_tasks:
   - 1 task = kompleksitas 1
   - 2-4 tasks = kompleksitas 2
   - 5-9 tasks = kompleksitas 3
   - 10-14 tasks = kompleksitas 4
   - 15+ tasks = kompleksitas 5
6. Return fullname in lowercase
7. Return project name in lowercase

Return ONLY valid JSON (no markdown, no explanation):
{{
  "data": [
    {{
      "fullname": "lowercase name",
      "project": "lowercase project",
      "start_date": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "finish_date": ["YYYY-MM-DD", "YYYY-MM-DD"],
      "total_tasks": 0,
      "duration_days": 0,
      "task": ["lowercase task 1", "lowercase task 2"],
      "kompleksitas": 0
    }}
  ]
}}"""

    try:
        print("📤 Sending to LLM...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 8192,
                    "num_ctx": 16384,
                    "repeat_penalty": 1.1,
                }
            },
            timeout=300
        )

        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}")
            return None

        raw = response.json().get("response", "").strip()

        print(f"\n📋 LLM Output ({len(raw)} chars)")
        print("=" * 70)

        # Parse JSON
        try:
            cleaned = re.sub(r'```json\s*|\s*```', '', raw).strip()
            result = json.loads(cleaned)

            if "data" in result and isinstance(result["data"], list):
                # Post-process: ensure all data is correct
                result = post_process_result(result, json_data)
                print(f"✅ JSON extracted! Found {len(result['data'])} people")
                return result
            else:
                print("⚠️ Invalid JSON structure")
                return None

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================
# 🔧 Post-Process Result
# =============================
def post_process_result(result: Dict, original_data: Dict) -> Dict:
    """Post-process LLM result to ensure correctness"""

    tasks = original_data.get("tasks", [])

    for person in result.get("data", []):
        fullname = person.get("fullname", "").lower().strip()

        # Get all tasks for this person from original data
        person_tasks = [t for t in tasks if t.get(
            "resource", "").lower() == fullname]

        # Rebuild arrays correctly
        start_dates = []
        finish_dates = []
        task_names = []

        for task in person_tasks:
            start_dates.append(task.get("start_date"))
            finish_dates.append(task.get("finish_date"))
            task_names.append(task.get("task_name", "").lower())

        # Update person data
        person["start_date"] = start_dates
        person["finish_date"] = finish_dates
        person["task"] = task_names
        person["total_tasks"] = len(task_names)

        # Calculate duration correctly
        if start_dates and finish_dates:
            try:
                earliest_start = min(datetime.strptime(
                    d, "%Y-%m-%d") for d in start_dates)
                latest_finish = max(datetime.strptime(d, "%Y-%m-%d")
                                    for d in finish_dates)
                person["duration_days"] = (
                    latest_finish - earliest_start).days + 1
            except:
                person["duration_days"] = 0

        # Calculate kompleksitas correctly
        task_count = person["total_tasks"]
        if task_count >= 15:
            person["kompleksitas"] = 5
        elif task_count >= 10:
            person["kompleksitas"] = 4
        elif task_count >= 5:
            person["kompleksitas"] = 3
        elif task_count >= 2:
            person["kompleksitas"] = 2
        else:
            person["kompleksitas"] = 1

        # Ensure lowercase
        person["fullname"] = fullname
        person["project"] = person.get("project", "").lower()

    return result


# =============================
# 🔍 Fungsi: Verify
# =============================
def verify_llm_result(result: Dict, original_data: Dict) -> bool:
    """Verify if LLM result matches original data"""

    if not result or "data" not in result:
        return False

    # Create resource-task mapping from original data
    original_mapping = defaultdict(list)
    for task in original_data.get("tasks", []):
        resource = task.get("resource", "").lower().strip()
        if resource:
            original_mapping[resource].append(
                task.get("task_name", "").lower())

    # Check each person in result
    errors = []
    for person in result["data"]:
        fullname = person.get("fullname", "").lower()

        result_tasks = set(person.get("task", []))
        expected_tasks = set(original_mapping.get(fullname, []))

        if result_tasks != expected_tasks:
            missing = expected_tasks - result_tasks
            extra = result_tasks - expected_tasks

            if missing:
                errors.append(
                    f"❌ {fullname.upper()}: Missing {len(missing)} tasks:")
                for t in missing:
                    errors.append(f"   - {t}")
            if extra:
                errors.append(
                    f"❌ {fullname.upper()}: Extra {len(extra)} tasks:")
                for t in extra:
                    errors.append(f"   - {t}")

    if errors:
        print("\n⚠️ VERIFICATION ERRORS:")
        print("=" * 70)
        for error in errors:
            print(error)
        print("=" * 70)
        return False

    print("\n✅ VERIFICATION PASSED: All tasks correctly assigned!")
    return True


# =============================
# 📂 Load JSON File
# =============================
def load_json_file(file_path: str):
    """Load JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file")
        return None


# =============================
# 🎯 Main Execution
# =============================
if __name__ == "__main__":
    print("🚀 Project Task Extraction - Fixed LLM Version")
    print("=" * 70)
    print()

    # File input
    file_path = "../extract/Project Plan - DSE - OneCert Integration.json"

    # Check for available models
    model_options = ["llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"]
    model_name = None

    print("🔍 Checking available models...")
    for model in model_options:
        if check_ollama_model(model):
            model_name = model
            break

    if not model_name:
        print("\n❌ No suitable LLM model found!")
        print("💡 Install one of these:")
        for model in model_options:
            print(f"   - ollama pull {model}")
        exit(1)

    print(f"\n✅ Using model: {model_name}")
    print()

    # Load file
    print(f"📂 Loading: {file_path}")
    json_data = load_json_file(file_path)

    if not json_data:
        exit(1)

    print(
        f"✅ Loaded project: {json_data.get('project', {}).get('name', 'Unknown')}")
    print(f"✅ Total tasks: {len(json_data.get('tasks', []))}")

    # Show resource breakdown
    print("\n📊 Resource Breakdown:")
    resource_count = defaultdict(int)
    for task in json_data.get("tasks", []):
        resource = task.get("resource", "").strip()
        if resource:
            resource_count[resource] += 1

    for resource, count in sorted(resource_count.items()):
        print(f"   - {resource}: {count} tasks")
    print()

    # LLM extraction
    print(f"🧠 Extracting with LLM...")
    print("⏳ Please wait...")
    print()

    result = extract_with_llm(json_data, model_name)

    if not result:
        print("\n❌ LLM extraction failed!")
        exit(1)

    # Verify result
    verify_llm_result(result, json_data)

    # Display results
    print()
    print("=" * 70)
    print("📊 HASIL EKSTRAKSI:")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()

    # Summary per person
    print("👥 RINGKASAN PER ORANG:")
    print("=" * 70)

    for person in result["data"]:
        print(f"\n👤 {person.get('fullname', 'N/A').upper()}")
        print("-" * 70)
        print(f"   📌 Project: {person.get('project', 'N/A')}")
        print(
            f"   📅 Date Range: {person['start_date'][0]} → {person['finish_date'][-1]}")
        print(f"   ⏱️  Duration: {person.get('duration_days', 0)} days")
        print(f"   📋 Total Tasks: {person.get('total_tasks', 0)}")

        if person.get('task'):
            print(f"   🔄 Tasks:")
            for idx, task in enumerate(person['task'], 1):
                print(f"      {idx}. {task}")

        stars = "⭐" * person['kompleksitas']
        labels = {1: "Simple", 2: "Basic", 3: "Moderate",
                  4: "Complex", 5: "Very Complex"}
        label = labels.get(person['kompleksitas'], "Unknown")
        print(
            f"   🎯 Kompleksitas: {stars} {person['kompleksitas']}/5 - {label}")

    print()
    print("=" * 70)

    # Save
    output_file = "../../dashboard-project-plan/data/projects.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to: {output_file}")
    print("✅ Process complete!")