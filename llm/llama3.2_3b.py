import json
import requests
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime

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
# 🧠 Fungsi: Ekstraksi dengan LLM
# =============================
def extract_with_llm(text: str, model: str = "llama3.2:1b"):
    """Ekstraksi menggunakan LLM - format array of objects"""
    
    # Ambil bagian penting (lebih banyak untuk konteks)
    lines = text.split('\n')
    important_text = '\n'.join(lines[:400])
    
    # Prompt yang SANGAT SIMPLE dan JELAS
    prompt = f"""You must return ONLY valid JSON. No explanation, no markdown.

Extract people from this project document and create JSON array.

Required format:
{{"data":[{{"fullname":"name","project":"proj","start_date":"MM/DD/YY","end_date":"MM/DD/YY","total_tasks":0,"duration_days":0,"phases":["PHASE1"],"kompleksitas":0}}]}}

IMPORTANT:
- phases: ONLY use main phase names: ANALYSIS, DESIGN, DEVELOPMENT, TESTING, DEPLOYMENT
- DO NOT include task names like "FSD Sign Off" or "Outgoing - Data Entry"
- Keep phases array SHORT (1-3 items max)

Find people: Herman, Ryan, Alvyn, Jeannie, Slamet, QA
- fullname: lowercase
- project: from "Project:" line (e.g., "QNB - QFT MX to MT")
- dates: MM/DD/YY format
- total_tasks: count their appearances
- phases: ONLY main phases, not task names
- kompleksitas: 1-5 rating

Document:
{important_text}

JSON:"""

    try:
        print("📤 Sending to LLM...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # PENTING: Force JSON output
                "options": {
                    "temperature": 0.0,  # Sangat deterministik
                    "top_p": 1.0,
                    "num_predict": 4096,  # DINAIKKAN! Was 2500, now 4096
                    "num_ctx": 16000,     # Context window lebih besar
                }
            },
            # timeout=300  # 5 menit untuk safety
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP Error {response.status_code}")
            return None
            
        raw = response.json().get("response", "").strip()
        
        print(f"\n📋 LLM Output ({len(raw)} chars):")
        print("=" * 70)
        print(raw[:1000] if len(raw) > 1000 else raw)
        if len(raw) > 1000:
            print(f"... (showing first 1000 of {len(raw)} chars)")
        print("=" * 70)
        print()
        
        # Check if JSON is truncated
        if len(raw) >= 8000 and not raw.strip().endswith('}'):
            print("⚠️ Warning: Output might be truncated (too long)")
        
        # Direct parse (karena format="json" seharusnya sudah return valid JSON)
        try:
            result = json.loads(raw)
            if "data" in result and isinstance(result["data"], list):
                print(f"✅ JSON extracted! Found {len(result['data'])} people")
                return result
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}")
            
            # Try to fix truncated JSON
            if "Unterminated string" in str(e):
                print("🔧 Attempting to fix truncated JSON...")
                # Find last complete person entry
                last_complete = raw.rfind('}},')
                if last_complete > 0:
                    # Truncate to last complete entry and close array
                    fixed = raw[:last_complete + 2] + ']}'
                    try:
                        result = json.loads(fixed)
                        if "data" in result:
                            print(f"✅ Fixed! Extracted {len(result['data'])} people (truncated)")
                            return result
                    except:
                        pass
        
        # Fallback: Extract dengan brace matching
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(raw):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = raw[start_idx:i+1]
                        result = json.loads(json_str)
                        if "data" in result:
                            print(f"✅ JSON extracted (brace matching)! Found {len(result['data'])} people")
                            return result
                    except:
                        continue
        
        # Last attempt: Clean markdown
        cleaned = re.sub(r'```json|```', '', raw).strip()
        try:
            result = json.loads(cleaned)
            if "data" in result:
                print(f"✅ JSON extracted (cleaned)! Found {len(result['data'])} people")
                return result
        except:
            pass
        
        print("⚠️ Failed to extract valid JSON with 'data' field")
        print(f"🔍 Raw output length: {len(raw)} chars")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# =============================
# 🔍 Fungsi: Ekstraksi Manual per Person
# =============================
def extract_manual_per_person(text: str) -> Dict:
    """Ekstraksi manual - return array format"""
    
    # Extract project name (clean)
    project_name = ""
    project_patterns = [
        r'Project:\s*([A-Za-z0-9\s\-\&]+?)(?=\s+Date:)',
        r'Project:\s*([A-Za-z0-9\s\-\&/]+?)(?=\s*\|)',
    ]
    
    for pattern in project_patterns:
        match = re.search(pattern, text)
        if match:
            project_name = match.group(1).strip()
            # Clean whitespace
            project_name = re.sub(r'\s+', ' ', project_name)
            # Remove "MXto" typo -> "MX to"
            project_name = project_name.replace('MXto', 'MX to')
            if len(project_name) > 5:
                break
    
    # Extract all valid dates (MM/DD/YY format)
    all_dates = re.findall(r'\b(\d{1,2}/\d{1,2}/\d{2})\b', text)
    valid_dates = []
    
    for date in all_dates:
        try:
            parts = date.split('/')
            m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
            if 1 <= m <= 12 and 1 <= d <= 31:
                valid_dates.append(date)
        except:
            continue
    
    overall_start = valid_dates[0] if valid_dates else ""
    overall_end = valid_dates[-1] if valid_dates else ""
    
    # Calculate overall duration
    overall_duration = 0
    if overall_start and overall_end:
        try:
            start_dt = datetime.strptime(overall_start, "%m/%d/%y")
            end_dt = datetime.strptime(overall_end, "%m/%d/%y")
            overall_duration = abs((end_dt - start_dt).days)
        except:
            pass
    
    # Extract phases
    all_phases = []
    phase_keywords = ['ANALYSIS', 'DESIGN', 'DEVELOPMENT', 'TESTING', 'DEPLOYMENT']
    for keyword in phase_keywords:
        if keyword in text.upper():
            all_phases.append(keyword)
    
    # Find all people
    common_names = ['Herman', 'Ryan', 'Alvyn', 'Jeannie', 'Slamet', 'QA']
    data_array = []
    
    # Parse document to find tasks per person
    lines = text.split('\n')
    
    for name in common_names:
        if name not in text:
            continue
        
        # Count tasks where this person appears
        task_count = 0
        person_phases = set()
        person_dates = []
        
        for line in lines:
            if name in line:
                task_count += 1
                
                # Find dates in this line
                line_dates = re.findall(r'\b(\d{1,2}/\d{1,2}/\d{2})\b', line)
                person_dates.extend([d for d in line_dates if d in valid_dates])
                
                # Find phases in this line
                for phase in phase_keywords:
                    if phase in line.upper():
                        person_phases.add(phase)
        
        # Get person's date range
        if person_dates:
            person_start = min(person_dates)
            person_end = max(person_dates)
        else:
            person_start = overall_start
            person_end = overall_end
        
        # Calculate duration for this person
        duration_days = 0
        if person_start and person_end:
            try:
                start_dt = datetime.strptime(person_start, "%m/%d/%y")
                end_dt = datetime.strptime(person_end, "%m/%d/%y")
                duration_days = abs((end_dt - start_dt).days)
            except:
                duration_days = overall_duration
        
        # Use all phases if no specific phases found
        if not person_phases:
            person_phases = set(all_phases)
        
        # Calculate complexity based on task count
        if task_count >= 20:
            kompleksitas = 5
        elif task_count >= 15:
            kompleksitas = 4
        elif task_count >= 10:
            kompleksitas = 3
        elif task_count >= 5:
            kompleksitas = 2
        else:
            kompleksitas = 1
        
        # Create person object
        person_obj = {
            "fullname": name.lower(),
            "project": project_name,
            "start_date": person_start,
            "end_date": person_end,
            "total_tasks": task_count,
            "duration_days": duration_days,
            "phases": sorted(list(person_phases)),
            "kompleksitas": kompleksitas
        }
        
        data_array.append(person_obj)
    
    return {"data": data_array}


# =============================
# 📂 Load OCR File
# =============================
def load_ocr_file(file_path: str):
    """Load JSON OCR result"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        ocr_text = data.get("combined_text") or data.get("docling_text", "")
        if not ocr_text:
            print("⚠️ No text found in JSON")
            return None
        
        return ocr_text
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
    print("🚀 OCR Project Extraction - Array Format")
    print("=" * 70)
    print()
    
    file_path = "../extract/QNB - QFT MX to MT - Project Plan Phase 2 v.05.1 (3)_extracted.json"
    
    # Try models in order of preference
    model_options = ["llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"]
    model_name = None
    
    print("🔍 Checking available models...")
    for model in model_options:
        if check_ollama_model(model):
            model_name = model
            break
    
    if not model_name:
        print("\n⚠️ No suitable LLM model found!")
        print("💡 Install one of these:")
        for model in model_options:
            print(f"   - ollama pull {model}")
        print("\n🔄 Will use manual extraction only")
        use_llm = False
    else:
        use_llm = True
        print(f"\n✅ Using model: {model_name}")
    
    print()
    
    # Load file
    print(f"📂 Loading: {file_path}")
    ocr_text = load_ocr_file(file_path)
    
    if not ocr_text:
        exit(1)
    
    print(f"✅ Loaded: {len(ocr_text):,} chars")
    print()
    
    result = None
    
    # Try LLM (with retry)
    if use_llm:
        print("🧠 Extracting with LLM...")
        print("⏳ Wait 30-180 seconds...")
        print()
        
        # Try up to 2 times
        for attempt in range(2):
            if attempt > 0:
                print(f"\n🔄 Retry attempt {attempt + 1}...")
            
            result = extract_with_llm(ocr_text, model_name)
            
            if result and "data" in result and len(result["data"]) > 0:
                print(f"\n✅ LLM extraction successful on attempt {attempt + 1}!")
                break
            
            if attempt == 0:
                print("⚠️ First attempt failed, retrying with adjusted prompt...")
        
        # Validate result
        if result and "data" in result:
            # Post-process: ensure all fields are present and calculate total_tasks
            print("\n🔧 Post-processing data...")
            
            for person in result["data"]:
                # Ensure all fields exist
                if "fullname" not in person:
                    person["fullname"] = ""
                if "project" not in person:
                    person["project"] = ""
                if "start_date" not in person:
                    person["start_date"] = ""
                if "end_date" not in person:
                    person["end_date"] = ""
                if "duration_days" not in person:
                    person["duration_days"] = 0
                if "phases" not in person:
                    person["phases"] = []
                if "kompleksitas" not in person:
                    person["kompleksitas"] = 1
                
                # Calculate total_tasks from original text
                name = person["fullname"].title()  # "herman" -> "Herman"
                if name:
                    # Count occurrences in text (more accurate)
                    task_count = 0
                    lines = ocr_text.split('\n')
                    for line in lines:
                        # Count if name appears in a task line (has | separator)
                        if name in line and '|' in line:
                            task_count += 1
                    
                    person["total_tasks"] = task_count
                    print(f"   ✓ {name}: {task_count} tasks")
                else:
                    person["total_tasks"] = 0
            
            print("✅ Post-processing complete")
    
    # Fallback to manual
    if not result:
        print("\n🔄 Using manual extraction...")
        result = extract_manual_per_person(ocr_text)
        print("✅ Manual extraction done")
    
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
    
    if result and "data" in result:
        for person in result["data"]:
            print(f"\n👤 {person.get('fullname', 'N/A').title()}")
            print("-" * 70)
            print(f"   📌 Project: {person.get('project', 'N/A')}")
            print(f"   📅 Timeline: {person.get('start_date', 'N/A')} → {person.get('end_date', 'N/A')}")
            print(f"   ⏱️  Duration: {person.get('duration_days', 0)} hari")
            print(f"   📋 Total Tasks: {person.get('total_tasks', 0)}")
            
            if person.get('phases'):
                phases_str = ', '.join(person['phases'])
                print(f"   🔄 Phases: {phases_str}")
            
            if person.get('kompleksitas'):
                stars = "⭐" * person['kompleksitas']
                labels = {1:"Simple", 2:"Basic", 3:"Moderate", 4:"Complex", 5:"Very Complex"}
                label = labels.get(person['kompleksitas'], "Unknown")
                print(f"   🎯 Kompleksitas: {stars} {person['kompleksitas']}/5 - {label}")
    
    print()
    print("=" * 70)
    
    # Save
    output_file = "extracted_per_person.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to: {output_file}")