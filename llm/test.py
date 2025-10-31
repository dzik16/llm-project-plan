import json
import requests
import re
from datetime import datetime
from collections import defaultdict

def extract_with_ollama(ocr_text, model="llama3.2:1b"):
    """
    Gunakan Ollama LLM untuk ekstraksi data project secara intelligent
    """
    
    # Truncate text jika terlalu panjang untuk context window
    max_chars = 15000
    if len(ocr_text) > max_chars:
        ocr_text = ocr_text[:max_chars] + "\n... [text truncated]"
    
    prompt = f"""You are a project management data extraction expert. Analyze this project plan document and extract team member information.

DOCUMENT TEXT:
{ocr_text}

TASK:
Extract information for EACH person/resource mentioned in the document. Look for:
- Resource Names column (people assigned to tasks)
- Names appearing after dates or task descriptions
- Common names like: Herman, Alvyn, Ryan, Jeannie, Slamet, QA, Rendy, PG1, PG2, PG3, etc.

For EACH person found, provide:
1. fullname: lowercase name (e.g., "herman", "alvyn", "pg1")
2. project: extract from "Project:" line in document
3. start_date: earliest date they appear (format: MM/DD/YY)
4. end_date: latest date they appear (format: MM/DD/YY)
5. total_tasks: count how many times they appear in task assignments
6. duration_days: sum of all "X days" durations for their tasks
7. phases: list of phases they work in (ANALYSIS & DESIGN, DEVELOPMENT, TESTING, DEPLOYMENT, KICK OFF)
8. kompleksitas: 1-5 score based on task count and duration

IMPORTANT RULES:
- Detect phase from keywords in task names (analysis, design, development, testing, deployment, kick off)
- If someone appears multiple times, aggregate all their data
- Use exact date format: MM/DD/YY (e.g., "08/12/24" or "8/12/25")
- Skip header rows and non-person entries

Return ONLY valid JSON in this exact format:
{{
  "data": [
    {{
      "fullname": "name",
      "project": "Project Name",
      "start_date": "MM/DD/YY",
      "end_date": "MM/DD/YY",
      "total_tasks": number,
      "duration_days": number,
      "phases": ["PHASE1", "PHASE2"],
      "kompleksitas": number
    }}
  ]
}}

Return ONLY the JSON, no explanation."""

    try:
        print("🤖 Mengirim data ke Ollama LLM...")
        print(f"   Model: {model}")
        print(f"   Text length: {len(ocr_text)} chars")
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Low temperature for more consistent output
                    'num_predict': 2000
                }
            },
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_output = result.get('response', '')
            
            print("✓ Response diterima dari LLM")
            print(f"  Output length: {len(llm_output)} chars")
            
            # Parse JSON dari response
            try:
                # Coba ekstrak JSON dari response (kadang LLM menambahkan text)
                json_match = re.search(r'\{[\s\S]*"data"[\s\S]*\}', llm_output)
                if json_match:
                    json_str = json_match.group(0)
                    extracted_data = json.loads(json_str)
                else:
                    # Coba parse langsung
                    extracted_data = json.loads(llm_output)
                
                if 'data' in extracted_data:
                    print(f"✓ Berhasil ekstrak {len(extracted_data['data'])} records dari LLM")
                    return extracted_data
                else:
                    print("⚠️  Response tidak memiliki key 'data'")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing JSON: {e}")
                print(f"  LLM Output sample: {llm_output[:500]}...")
                return None
        else:
            print(f"✗ HTTP Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("✗ Tidak dapat terhubung ke Ollama")
        print("  Pastikan Ollama berjalan: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("✗ Request timeout (>180s)")
        print("  LLM mungkin memproses terlalu lama")
        return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def refine_with_post_processing(data, ocr_text):
    """
    Post-processing untuk memperbaiki dan validasi data dari LLM
    """
    if not data or 'data' not in data:
        return data
    
    refined = []
    
    # Ekstrak project name dari text
    project_match = re.search(r'Project:\s*([^\n]+?)(?:\s+Date:|$)', ocr_text)
    project_name = project_match.group(1).strip() if project_match else "Unknown Project"
    project_name = re.sub(r'\s+Pro\s*$', '', project_name)
    
    for person in data['data']:
        # Validate and clean data
        fullname = person.get('fullname', '').lower().strip()
        if not fullname or len(fullname) < 2:
            continue
        
        # Use extracted project name if not provided
        if not person.get('project') or person['project'] == 'Unknown Project':
            person['project'] = project_name
        
        # Validate dates
        start_date = person.get('start_date', '')
        end_date = person.get('end_date', '')
        
        # Normalize date format
        if start_date:
            start_date = normalize_date(start_date)
        if end_date:
            end_date = normalize_date(end_date)
        
        # Ensure complexity is 1-5
        kompleksitas = person.get('kompleksitas', 1)
        kompleksitas = max(1, min(5, kompleksitas))
        
        # Ensure phases is a list
        phases = person.get('phases', [])
        if not isinstance(phases, list):
            phases = [str(phases)] if phases else []
        
        refined.append({
            'fullname': fullname,
            'project': person.get('project', project_name),
            'start_date': start_date,
            'end_date': end_date,
            'total_tasks': person.get('total_tasks', 0),
            'duration_days': person.get('duration_days', 0),
            'phases': phases,
            'kompleksitas': kompleksitas
        })
    
    return {'data': refined}


def normalize_date(date_str):
    """Normalize date format to MM/DD/YY"""
    if not date_str:
        return ""
    
    # Already in correct format
    if re.match(r'\d{1,2}/\d{1,2}/\d{2}', date_str):
        parts = date_str.split('/')
        return f"{parts[0].zfill(2)}/{parts[1].zfill(2)}/{parts[2]}"
    
    return date_str


def extract_project_data_with_llm(json_file_path, output_file_path="output.json", model="llama3.2:1b"):
    """
    Main function: ekstrak data menggunakan Ollama LLM
    """
    
    print("📖 Membaca file JSON...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"✗ Error membaca file: {e}")
        return None
    
    # Ambil text dari hasil OCR
    ocr_text = data.get('combined_text', '') or data.get('docling_text', '')
    
    if not ocr_text:
        print("✗ Tidak ada text ditemukan dalam JSON")
        return None
    
    print(f"📄 Total {len(ocr_text)} karakter ditemukan")
    
    # Extract dengan LLM
    result = extract_with_ollama(ocr_text, model)
    
    if not result:
        print("\n⚠️  LLM extraction gagal, mencoba fallback method...")
        return fallback_extraction(ocr_text, output_file_path)
    
    # Post-processing
    print("\n🔧 Melakukan post-processing...")
    result = refine_with_post_processing(result, ocr_text)
    
    if not result or not result.get('data'):
        print("⚠️  Tidak ada data setelah post-processing")
        return None
    
    # Save output
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Data berhasil diekstrak untuk {len(result['data'])} orang")
    print(f"✅ Output disimpan di: {output_file_path}\n")
    
    return result


def fallback_extraction(ocr_text, output_file_path):
    """
    Fallback: ekstraksi regex sederhana jika LLM gagal
    """
    print("⚙️  Menggunakan ekstraksi regex fallback...")
    
    lines = ocr_text.split('\n')
    
    # Ekstrak project name
    project_match = re.search(r'Project:\s*([^\n]+?)(?:\s+Date:|$)', ocr_text)
    project_name = project_match.group(1).strip() if project_match else "Unknown Project"
    project_name = re.sub(r'\s+Pro\s*$', '', project_name)
    
    # Cari semua resource names
    resources = set()
    for line in lines:
        # Pattern: setelah tanggal dan predecessor
        matches = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d+\s+([A-Za-z0-9]+(?:\s*,\s*[A-Za-z0-9]+)*)', line)
        for match in matches:
            for name in match.split(','):
                name = name.strip().lower()
                if name and len(name) > 1 and len(name) < 20:
                    if not any(x in name for x in ['days', 'task', 'page', 'manual']):
                        resources.add(name)
    
    if not resources:
        print("✗ Tidak ada resource ditemukan dengan fallback method")
        return {'data': []}
    
    print(f"  Ditemukan {len(resources)} resources: {', '.join(sorted(resources))}")
    
    # Aggregate data per resource
    person_data = defaultdict(lambda: {
        'dates': [],
        'durations': [],
        'phases': set()
    })
    
    current_phase = None
    for line in lines:
        line_lower = line.lower()
        
        # Detect phase
        if 'analysis' in line_lower or 'design' in line_lower:
            current_phase = 'ANALYSIS & DESIGN'
        elif 'development' in line_lower:
            current_phase = 'DEVELOPMENT'
        elif 'testing' in line_lower:
            current_phase = 'TESTING'
        elif 'deployment' in line_lower:
            current_phase = 'DEPLOYMENT'
        elif 'kick off' in line_lower:
            current_phase = 'KICK OFF'
        
        for resource in resources:
            if resource in line_lower:
                dates = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', line)
                duration_match = re.search(r'(\d+(?:\.\d+)?)\s+days?', line)
                
                if dates:
                    person_data[resource]['dates'].extend(dates)
                if duration_match:
                    person_data[resource]['durations'].append(float(duration_match.group(1)))
                if current_phase:
                    person_data[resource]['phases'].add(current_phase)
    
    # Build result
    result_data = []
    for name, data in person_data.items():
        if not data['dates']:
            continue
        
        dates = [normalize_date(d) for d in data['dates']]
        dates = [d for d in dates if d]
        
        if dates:
            start_date = min(dates)
            end_date = max(dates)
        else:
            start_date = end_date = ""
        
        total_tasks = len(data['dates'])
        duration_days = int(sum(data['durations'])) if data['durations'] else total_tasks
        
        kompleksitas = min(5, max(1, (total_tasks + duration_days // 10) // 3))
        
        result_data.append({
            'fullname': name,
            'project': project_name,
            'start_date': start_date,
            'end_date': end_date,
            'total_tasks': total_tasks,
            'duration_days': duration_days,
            'phases': sorted(list(data['phases'])),
            'kompleksitas': kompleksitas
        })
    
    result = {'data': sorted(result_data, key=lambda x: x['fullname'])}
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Fallback extraction selesai: {len(result_data)} orang")
    return result


# Main execution
if __name__ == "__main__":
    import sys
    
    # Parse arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "../extract/QNB - QFT MX to MT - Project Plan Phase 2 v.05.1_extracted.json"
    
    output_file = "project_summary.json"
    
    # Optional: specify model
    model = "llama3.2:1b"
    if len(sys.argv) > 2:
        model = sys.argv[2]
    
    print("="*70)
    print("   PROJECT DATA EXTRACTOR - AI-POWERED WITH OLLAMA LLM")
    print("="*70)
    print(f"\n📁 Input file : {input_file}")
    print(f"💾 Output file: {output_file}")
    print(f"🤖 LLM Model  : {model}")
    print(f"\n{'='*70}\n")
    
    try:
        # Extract dengan LLM
        result = extract_project_data_with_llm(input_file, output_file, model)
        
        if result and result['data']:
            print("="*70)
            print("HASIL EKSTRAKSI:")
            print("="*70)
            
            for person in result['data']:
                print(f"\n👤 {person['fullname'].upper()}")
                print(f"   Project    : {person['project']}")
                print(f"   Tasks      : {person['total_tasks']}")
                print(f"   Duration   : {person['duration_days']} days")
                print(f"   Period     : {person['start_date']} - {person['end_date']}")
                print(f"   Phases     : {', '.join(person['phases']) if person['phases'] else 'N/A'}")
                print(f"   Complexity : {'⭐' * person['kompleksitas']} ({person['kompleksitas']}/5)")
            
            print("\n" + "="*70)
            print(f"✅ SUCCESS! Total {len(result['data'])} orang berhasil diekstrak.")
            print("="*70)
            print(f"\n💡 Tips:")
            print(f"   • Gunakan file lain: python {sys.argv[0]} <path_to_json>")
            print(f"   • Gunakan model lain: python {sys.argv[0]} <file> llama3.2:3b")
            print(f"   • Model tersedia: llama3.2:1b, llama3.2:3b, llama3.1:8b, dll")
        else:
            print("\n❌ Ekstraksi gagal atau tidak ada data yang ditemukan!")
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"\n❌ Error: File '{input_file}' tidak ditemukan!")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Proses dibatalkan oleh user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)