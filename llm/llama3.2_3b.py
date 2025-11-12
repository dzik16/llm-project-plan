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
    processed_hashes = {f["hash"] for f in processed_data.get("processed_files", [])}
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
        person_tasks = [t for t in tasks if t.get("resource", "").lower() == fullname]

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
                earliest_start = min(datetime.strptime(d, "%Y-%m-%d") for d in start_dates)
                latest_finish = max(datetime.strptime(d, "%Y-%m-%d") for d in finish_dates)
                person["duration_days"] = (latest_finish - earliest_start).days + 1
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
# 🔀 Fungsi: Merge Data
# =============================

def merge_project_data(existing_data: Dict, new_data: Dict) -> Dict:
    """Merge data baru dengan data yang sudah ada"""
    
    if not existing_data or "data" not in existing_data:
        return new_data
    
    if not new_data or "data" not in new_data:
        return existing_data
    
    # Create a dictionary with (fullname, project) as key
    merged = {}
    
    # Add existing data
    for person in existing_data.get("data", []):
        key = (person.get("fullname", "").lower(), person.get("project", "").lower())
        merged[key] = person
    
    # Add or merge new data
    for person in new_data.get("data", []):
        key = (person.get("fullname", "").lower(), person.get("project", "").lower())
        
        if key in merged:
            # Merge if same person and project (shouldn't happen, but just in case)
            existing = merged[key]
            
            # Merge dates
            all_start = list(set(existing.get("start_date", []) + person.get("start_date", [])))
            all_finish = list(set(existing.get("finish_date", []) + person.get("finish_date", [])))
            
            existing["start_date"] = sorted(all_start)
            existing["finish_date"] = sorted(all_finish)
            
            # Merge tasks
            all_tasks = list(set(existing.get("task", []) + person.get("task", [])))
            existing["task"] = sorted(all_tasks)
            existing["total_tasks"] = len(all_tasks)
            
            # Recalculate duration
            if existing["start_date"] and existing["finish_date"]:
                try:
                    earliest = min(datetime.strptime(d, "%Y-%m-%d") for d in existing["start_date"])
                    latest = max(datetime.strptime(d, "%Y-%m-%d") for d in existing["finish_date"])
                    existing["duration_days"] = (latest - earliest).days + 1
                except:
                    pass
            
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
            # Add new entry
            merged[key] = person
    
    return {"data": list(merged.values())}


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
        print(f"💡 Total files tracked: {len(processed_data.get('processed_files', []))}")
        exit(0)

    print(f"📋 Files to process: {len(unprocessed_files)}")
    for f in unprocessed_files:
        print(f"   - {f['name']}")
    print()

    # Load existing results
    existing_results = {"data": []}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            print(f"📊 Loaded existing results: {len(existing_results.get('data', []))} entries")
        except:
            print("⚠️ Could not load existing results, starting fresh")
    else:
        print("📝 No existing results, starting fresh")
    print()

    # Process each unprocessed file
    all_new_data = {"data": []}
    processed_count = 0

    for idx, file_info in enumerate(unprocessed_files, 1):
        print(f"🔄 Processing [{idx}/{len(unprocessed_files)}]: {file_info['name']}")
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
            # Merge with accumulated new data
            all_new_data = merge_project_data(all_new_data, result)
            
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

    # Merge with existing results
    if processed_count > 0:
        print("=" * 70)
        print("🔀 Merging results...")
        final_results = merge_project_data(existing_results, all_new_data)
        
        print(f"   Previous entries: {len(existing_results.get('data', []))}")
        print(f"   New entries: {len(all_new_data.get('data', []))}")
        print(f"   Total entries: {len(final_results.get('data', []))}")
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
        
        # Group by person
        person_projects = defaultdict(list)
        for entry in final_results.get("data", []):
            person_projects[entry.get("fullname", "unknown")].append(entry)
        
        for person, projects in sorted(person_projects.items()):
            print(f"\n👤 {person.upper()}")
            print("-" * 70)
            total_tasks = sum(p.get("total_tasks", 0) for p in projects)
            print(f"   📋 Total projects: {len(projects)}")
            print(f"   📋 Total tasks: {total_tasks}")
            
            for project in projects:
                print(f"   📌 {project.get('project', 'unknown')}: {project.get('total_tasks', 0)} tasks")
        
        print()
        print("=" * 70)
        print(f"✅ Successfully processed {processed_count} new file(s)!")
        print(f"📊 Total entries in database: {len(final_results.get('data', []))}")
    else:
        print("❌ No files were successfully processed")

    print("=" * 70)
    print("✅ Process complete!")