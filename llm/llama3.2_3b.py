import json
import requests
import re
import os
import hashlib
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
# 📂 Fungsi: Scan JSON Files
# =============================

def get_file_hash(file_path: str) -> str:
    """Generate hash untuk file untuk tracking"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def scan_json_files(folder_path: str) -> List[Dict]:
    """Scan semua file JSON di folder"""
    json_files = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder tidak ditemukan: {folder_path}")
        return []

    for file_path in folder.glob("*.json"):
        try:
            file_hash = get_file_hash(str(file_path))
            json_files.append({
                "path": str(file_path),
                "name": file_path.name,
                "hash": file_hash
            })
        except Exception as e:
            print(f"⚠️ Error reading {file_path.name}: {e}")

    return json_files


def load_processed_files(tracking_file: str) -> Dict:
    """Load daftar file yang sudah diproses"""
    if os.path.exists(tracking_file):
        try:
            with open(tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"processed_files": []}
    return {"processed_files": []}


def save_processed_files(tracking_file: str, processed_data: Dict):
    """Simpan daftar file yang sudah diproses"""
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)


def get_unprocessed_files(json_files: List[Dict], processed_data: Dict) -> List[Dict]:
    """Dapatkan file yang belum diproses"""
    processed_hashes = {f["hash"]
                        for f in processed_data.get("processed_files", [])}
    return [f for f in json_files if f["hash"] not in processed_hashes]


# =============================
# 🧠 Fungsi: Ekstraksi dengan LLM
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
3. start_date: earliest start date (single date string, not array)
4. finish_date: latest finish date (single date string, not array)
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
      "start_date": "YYYY-MM-DD",
      "finish_date": "YYYY-MM-DD",
      "total_tasks": 0,
      "tasks": ["lowercase task 1", "lowercase task 2"],
      "kompleksitas": 0
    }}
  ]
}}"""

    try:
        print("   📤 Sending to LLM...")
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
            print(f"   ❌ HTTP Error {response.status_code}")
            return None

        raw = response.json().get("response", "").strip()

        # Parse JSON
        try:
            cleaned = re.sub(r'```json\s*|\s*```', '', raw).strip()
            result = json.loads(cleaned)

            if "data" in result and isinstance(result["data"], list):
                # Post-process: ensure all data is correct
                result = post_process_result(result, json_data)
                print(f"   ✅ Extracted {len(result['data'])} people")
                return result
            else:
                print("   ⚠️ Invalid JSON structure")
                return None

        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON decode error: {e}")
            return None

    except Exception as e:
        print(f"   ❌ Error: {e}")
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

        # Rebuild data correctly
        start_dates = []
        finish_dates = []
        task_names = []

        for task in person_tasks:
            start_dates.append(task.get("start_date"))
            finish_dates.append(task.get("finish_date"))
            task_names.append(task.get("task_name", "").lower())

        # Get earliest start and latest finish
        earliest_start = ""
        latest_finish = ""

        if start_dates and finish_dates:
            try:
                earliest_start = min(start_dates)
                latest_finish = max(finish_dates)
            except:
                earliest_start = start_dates[0] if start_dates else ""
                latest_finish = finish_dates[-1] if finish_dates else ""

        # Update person data
        person["start_date"] = earliest_start
        person["finish_date"] = latest_finish
        person["tasks"] = task_names
        person["total_tasks"] = len(task_names)

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
# 🔀 Fungsi: Merge & Group Data by Person
# =============================

def merge_and_group_by_person(existing_data: Dict, new_data: Dict) -> Dict:
    """Merge data baru dengan data existing dan group by person"""

    # Temporary storage: flat list with (fullname, project) as unique combo
    temp_data = []

    # Add existing data (convert from grouped format if needed)
    if existing_data and "people" in existing_data:
        for person in existing_data.get("people", []):
            fullname = person.get("fullname", "").lower()
            for project in person.get("projects", []):
                temp_data.append({
                    "fullname": fullname,
                    "project": project.get("project", "").lower(),
                    "start_date": project.get("start_date", ""),
                    "finish_date": project.get("finish_date", ""),
                    "total_tasks": project.get("total_tasks", 0),
                    "tasks": project.get("tasks", []),
                    "kompleksitas": project.get("kompleksitas", 0)
                })

    # Add new data
    if new_data and "data" in new_data:
        for person in new_data.get("data", []):
            temp_data.append({
                "fullname": person.get("fullname", "").lower(),
                "project": person.get("project", "").lower(),
                "start_date": person.get("start_date", ""),
                "finish_date": person.get("finish_date", ""),
                "total_tasks": person.get("total_tasks", 0),
                "tasks": person.get("tasks", []),
                "kompleksitas": person.get("kompleksitas", 0)
            })

    # Merge duplicates (same person + same project)
    merged_dict = {}
    for entry in temp_data:
        key = (entry["fullname"], entry["project"])

        if key in merged_dict:
            existing = merged_dict[key]

            # Merge tasks
            all_tasks = list(set(existing["tasks"] + entry["tasks"]))
            existing["tasks"] = sorted(all_tasks)
            existing["total_tasks"] = len(all_tasks)

            # Get earliest start and latest finish
            dates = [existing["start_date"], entry["start_date"]]
            dates = [d for d in dates if d]
            if dates:
                existing["start_date"] = min(dates)

            dates = [existing["finish_date"], entry["finish_date"]]
            dates = [d for d in dates if d]
            if dates:
                existing["finish_date"] = max(dates)

            # Recalculate kompleksitas
            task_count = existing["total_tasks"]
            if task_count >= 15:
                existing["kompleksitas"] = 5
            elif task_count >= 10:
                existing["kompleksitas"] = 4
            elif task_count >= 5:
                existing["kompleksitas"] = 3
            elif task_count >= 2:
                existing["kompleksitas"] = 2
            else:
                existing["kompleksitas"] = 1
        else:
            merged_dict[key] = entry

    # Group by person
    people_dict = defaultdict(list)
    for entry in merged_dict.values():
        fullname = entry["fullname"]
        project_data = {
            "project": entry["project"],
            "start_date": entry["start_date"],
            "finish_date": entry["finish_date"],
            "total_tasks": entry["total_tasks"],
            "kompleksitas": entry["kompleksitas"],
            "tasks": entry["tasks"]
        }
        people_dict[fullname].append(project_data)

    # Convert to final format
    people_list = []
    for fullname, projects in sorted(people_dict.items()):
        people_list.append({
            "fullname": fullname,
            "projects": sorted(projects, key=lambda x: x["project"])
        })

    return {"people": people_list}


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
        print(f"   ❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"   ❌ Invalid JSON file")
        return None


# =============================
# 🎯 Main Execution
# =============================

if __name__ == "__main__":
    print("🚀 Project Task Extraction - Batch Processing")
    print("=" * 70)
    print()

    # Configuration
    input_folder = "../extract"
    output_file = "../../dashboard-project-plan/data/projects.json"
    tracking_file = ".processed_files.json"

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

    print(f"✅ Using model: {model_name}")
    print()

    # Scan JSON files
    print(f"📂 Scanning folder: {input_folder}")
    json_files = scan_json_files(input_folder)

    if not json_files:
        print("❌ No JSON files found!")
        exit(1)

    print(f"✅ Found {len(json_files)} JSON files")
    print()

    # Load processed files tracking
    processed_data = load_processed_files(tracking_file)
    unprocessed_files = get_unprocessed_files(json_files, processed_data)

    if not unprocessed_files:
        print("✅ All files have been processed!")
        print(
            f"💡 Total files tracked: {len(processed_data.get('processed_files', []))}")
        exit(0)

    print(f"📋 Files to process: {len(unprocessed_files)}")
    for f in unprocessed_files:
        print(f"   - {f['name']}")
    print()

    # Load existing results
    existing_results = {"people": []}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # Count total entries
            total_entries = sum(len(p.get("projects", [])) for p in existing_results.get("people", []))
            print(f"📊 Loaded existing results: {len(existing_results.get('people', []))} people, {total_entries} project entries")
        except:
            print("⚠️ Could not load existing results, starting fresh")
    else:
        print("📝 No existing results, starting fresh")
    print()

    # Process each unprocessed file
    all_new_data = {"data": []}
    processed_count = 0

    for idx, file_info in enumerate(unprocessed_files, 1):
        print(
            f"🔄 Processing [{idx}/{len(unprocessed_files)}]: {file_info['name']}")
        print("-" * 70)

        # Load file
        json_data = load_json_file(file_info['path'])

        if not json_data:
            print(f"   ⚠️ Skipping due to load error")
            print()
            continue

        project_name = json_data.get('project', {}).get('name', 'Unknown')
        total_tasks = len(json_data.get('tasks', []))

        print(f"   📌 Project: {project_name}")
        print(f"   📋 Total tasks: {total_tasks}")

        # Extract with LLM
        result = extract_with_llm(json_data, model_name)

        if result:
            # Accumulate new data (still in flat format)
            for person in result.get("data", []):
                all_new_data["data"].append(person)

            # Mark as processed
            processed_data["processed_files"].append({
                "name": file_info['name'],
                "hash": file_info['hash'],
                "processed_at": datetime.now().isoformat(),
                "project": project_name
            })
            processed_count += 1
            print(f"   ✅ Successfully processed!")
        else:
            print(f"   ❌ Failed to process")

        print()

    # Merge with existing results and group by person
    if processed_count > 0:
        print("=" * 70)
        print("🔀 Merging and grouping results by person...")
        final_results = merge_and_group_by_person(existing_results, all_new_data)

        prev_people = len(existing_results.get('people', []))
        prev_entries = sum(len(p.get("projects", [])) for p in existing_results.get("people", []))
        new_entries = len(all_new_data.get('data', []))
        final_people = len(final_results.get('people', []))
        final_entries = sum(len(p.get("projects", [])) for p in final_results.get("people", []))

        print(f"   Previous: {prev_people} people, {prev_entries} project entries")
        print(f"   New: {new_entries} project entries")
        print(f"   Final: {final_people} people, {final_entries} project entries")
        print()

        # Save results
        print(f"💾 Saving to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # Save processed files tracking
        save_processed_files(tracking_file, processed_data)
        print(f"💾 Updated tracking file: {tracking_file}")
        print()

        # Display summary
        print("=" * 70)
        print("📊 SUMMARY BY PERSON:")
        print("=" * 70)

        for person in final_results.get("people", []):
            fullname = person.get("fullname", "unknown")
            projects = person.get("projects", [])
            total_tasks = sum(p.get("total_tasks", 0) for p in projects)

            print(f"\n👤 {fullname.upper()}")
            print("-" * 70)
            print(f"   📋 Total projects: {len(projects)}")
            print(f"   📋 Total tasks: {total_tasks}")

            for project in projects:
                print(f"   📌 {project.get('project', 'unknown')}: {project.get('total_tasks', 0)} tasks")

        print()
        print("=" * 70)
        print(f"✅ Successfully processed {processed_count} new file(s)!")
        print(f"📊 Total people in database: {final_people}")
        print(f"📊 Total project entries: {final_entries}")
    else:
        print("❌ No files were successfully processed")

    print("=" * 70)
    print("✅ Process complete!")