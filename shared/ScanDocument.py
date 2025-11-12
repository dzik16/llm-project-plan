#!/usr/bin/env python3
"""
project_extractor.py

Enhanced Project Plan extractor from PDF (text-based).
Only extracts tasks with resource names assigned.
Splits tasks with multiple resources into separate entries.
Skips files with no valid resource assignments.

Usage:
  python project_extractor.py /path/to/project_plan.pdf
  python project_extractor.py --dir /mnt/data
"""

import re
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import pdfplumber
import os

# ------------------------------
# Utility helpers
# ------------------------------
DATE_FORMAT_TRIES = [
    "%m/%d/%Y", "%m/%d/%y",
    "%Y-%m-%d",
    "%b %d, '%y", "%b %d, %Y", "%B %d, %Y",
    "%a %m/%d/%y", "%A %m/%d/%y",
    "%a %b %d, %Y", "%A %b %d, %Y",
    "%a %m/%d/%Y", "%A %m/%d/%Y",
]


def try_parse_date(s: str) -> Optional[str]:
    s = str(s).strip()
    if not s:
        return None

    # remove trailing question marks or extra punctuation
    s = re.sub(r"[?•\u2022]+$", "", s).strip()
    # remove weekday prefix like "Fri " or "Thu "
    s = re.sub(r'^[A-Za-z]{3,}\s+', '', s)

    # some PDF outputs include formats like "Jul 20, '25" -> normalize apostrophe
    s = s.replace("'", "'").replace("`", "'")

    for fmt in DATE_FORMAT_TRIES:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Try to catch month day year without comma: "Jul 20 25" or "Jul 20 2025"
    m = re.match(r'([A-Za-z]+)\s+(\d{1,2})[,\s]*\'?(\d{2,4})', s)
    if m:
        mon, day, yr = m.groups()
        try:
            yr = int(yr)
            if yr < 100:  # '25 -> 2025
                yr += 2000
            dt = datetime.strptime(f"{mon} {day} {yr}", "%b %d %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Try mm/dd with 2-digit year w/o leading zeros
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})$', s)
    if m:
        mm, dd, yy = m.groups()
        if len(yy) == 2:
            yy = int(yy) + 2000
        try:
            dt = datetime.strptime(f"{mm}/{dd}/{yy}", "%m/%d/%Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


def is_valid_resource_name(token: str) -> bool:
    """Check if token is a valid resource name (not a date, number, or calendar label)"""
    token = token.strip()
    if not token:
        return False

    # Exclude pure numbers
    if token.isdigit():
        return False

    # Exclude date patterns
    if re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', token):
        return False

    # Exclude month patterns
    if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', token, re.IGNORECASE):
        return False

    # Exclude weekday patterns
    if re.match(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', token, re.IGNORECASE):
        return False

    # Exclude calendar day labels
    if re.match(r'^[SMTWRF]{1,2}(\s+|$)', token):
        return False

    # Exclude common non-resource tokens
    exclude_tokens = ['page', 'task', 'external', 'tasks', 'manual', 'finish-only',
                      'split', 'milestone', 'duration-only', 'deadline', 'project',
                      'summary', 'inactive', 'rollup', 'progress', 'start-only',
                      'date', 'finish', 'start', 'duration', 'predecessor', 'id']
    if token.lower() in exclude_tokens:
        return False

    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', token):
        return False

    # Must be at least 2 characters
    if len(token) < 2:
        return False

    return True

# ------------------------------
# Main Extractor
# ------------------------------


class ProjectPlanExtractor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.project_name: Optional[str] = None

    def extract_from_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract project data from PDF.
        Returns None if no tasks with resources are found.
        """
        text = self._extract_text(pdf_path)
        if self.debug:
            print("=== RAW TEXT PREVIEW ===")
            print(text[:3000])  # First 3000 chars
            print("========================")

        self.project_name = self._extract_project_name(
            text) or os.path.splitext(os.path.basename(pdf_path))[0]
        tasks = self._parse_text_to_tasks(text)

        # Filter: only keep tasks with resources
        tasks_with_resources = [t for t in tasks if t.get(
            "resources") and len(t["resources"]) > 0]

        if not tasks_with_resources:
            print(f"⚠️  No tasks with resources found in this PDF")
            print(f"   Total tasks scanned: {len(tasks)}")
            print(f"   Tasks with resources: 0")
            return None

        # Split tasks: one task per resource
        expanded_tasks = self._expand_tasks_by_resource(tasks_with_resources)

        # Final check: ensure all expanded tasks have non-empty resource
        expanded_tasks = [
            t for t in expanded_tasks if t.get("resource", "").strip()]

        if not expanded_tasks:
            print(f"⚠️  No valid tasks after expansion")
            return None

        return {
            "project": {
                "name": self.project_name,
                "total_tasks": len(expanded_tasks),
                "total_tasks_scanned": len(tasks),
                "original_tasks_with_resources": len(tasks_with_resources),
                "extracted_at": datetime.now().isoformat()
            },
            "tasks": expanded_tasks
        }

    def _extract_text(self, pdf_path: str) -> str:
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                parts.append(page_text)
                if self.debug:
                    print(f"Extracted page {i+1}: {len(page_text)} chars")
        return "\n".join(parts)

    def _extract_project_name(self, text: str) -> Optional[str]:
        # Common header: "Project: <name>"
        for line in text.splitlines():
            m = re.search(r'Project:\s*(.+)', line, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                # Clean up the name - remove "Date:" suffix if present
                name = re.sub(r'\s+Date:.*$', '', name)
                if name:
                    return name
        return None

    def _expand_tasks_by_resource(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand tasks so that each task with multiple resources becomes multiple tasks,
        one per resource. Skip tasks with empty resources.
        """
        expanded = []

        for task in tasks:
            resources = task.get("resources", [])

            # Filter out empty resources
            resources = [r for r in resources if r and r.strip()]

            if not resources:
                if self.debug:
                    print(f"  Skipping task {task['id']}: no valid resources")
                continue

            if len(resources) == 1:
                # Single resource - keep as is but change to string
                task_copy = task.copy()
                task_copy["resource"] = resources[0]
                del task_copy["resources"]
                expanded.append(task_copy)
            else:
                # Multiple resources - create one task per resource
                for resource in resources:
                    if not resource or not resource.strip():
                        continue

                    task_copy = {
                        "id": task["id"],
                        "task_name": task["task_name"],
                        "duration": task["duration"].copy(),
                        "start_date": task["start_date"],
                        "finish_date": task["finish_date"],
                        "predecessors": task["predecessors"].copy() if task["predecessors"] else [],
                        "resource": resource
                    }
                    expanded.append(task_copy)

                    if self.debug:
                        print(
                            f"  Expanded task {task['id']} for resource: {resource}")

        return expanded

    def _parse_text_to_tasks(self, text: str) -> List[Dict[str, Any]]:
        lines = [ln for ln in text.splitlines()]
        tasks: List[Dict[str, Any]] = []

        # Find where task data starts - look for lines starting with digit
        start_idx = 0
        for i, ln in enumerate(lines):
            # Skip header lines
            if re.search(r'\bID\b.*\bTask\s*Name\b', ln, re.IGNORECASE):
                start_idx = i + 1
                if self.debug:
                    print(
                        f"Found header at line {i}, starting parse from line {start_idx}")
                break

        for line_num, ln in enumerate(lines[start_idx:], start=start_idx):
            if not ln.strip():
                continue

            # Skip page footers and non-task lines
            if (ln.strip().startswith("Page") or
                'Project:' in ln or
                re.search(r'\bTask\s+External\s+Tasks\b', ln) or
                re.search(r'\bMilestone\s+Inactive\b', ln) or
                    re.search(r'^\d+\s+\d+\s+\d+\s+\d+', ln)):  # Skip date header rows
                continue

            # Look for lines starting with task ID (1-3 digits followed by space and text)
            m = re.match(r'^(\d{1,3})\s+(.+)$', ln)
            if not m:
                continue

            task_id = m.group(1)
            remainder = m.group(2).strip()

            if self.debug:
                print(f"\n--- Line {line_num}: ID={task_id} ---")
                print(f"Remainder: {remainder[:200]}")

            # Parse the task line
            task = self._parse_task_line(task_id, remainder)

            if task:
                tasks.append(task)
                if self.debug:
                    print(f"✓ Parsed: {task['task_name']}")
                    print(f"  Resources: {task.get('resources', [])}")

        return tasks

    def _parse_task_line(self, task_id: str, remainder: str) -> Optional[Dict[str, Any]]:
        """
        Parse a task line with improved pattern matching.
        Format: <task_name> <duration> <start_date> <finish_date> [<predecessor>] <resources>

        Example: "Requirement Gathering II 4 days Wed 6/12/24 Wed 6/19/24 Herman Herman"
        """

        # Pattern to match: text, then duration, then two dates, then remaining tokens
        # Duration pattern: number + "day" or "days"
        pattern = r'^(.+?)\s+([\d.]+\s+days?\??)\s+((?:[A-Za-z]{3}\s+)?\d{1,2}/\d{1,2}/\d{2,4})\s+((?:[A-Za-z]{3}\s+)?\d{1,2}/\d{1,2}/\d{2,4})\s*(.*)$'

        m = re.match(pattern, remainder, re.IGNORECASE)

        if not m:
            if self.debug:
                print(f"  ✗ Pattern didn't match")
            return None

        task_name = m.group(1).strip()
        duration = m.group(2).strip()
        start_date = m.group(3).strip()
        finish_date = m.group(4).strip()
        remaining = m.group(5).strip()

        if self.debug:
            print(f"  Task: {task_name}")
            print(f"  Duration: {duration}")
            print(f"  Start: {start_date}")
            print(f"  Finish: {finish_date}")
            print(f"  Remaining: {remaining}")

        # Parse remaining tokens for predecessors and resources
        predecessors = []
        resources = []

        if remaining:
            # Split by whitespace
            tokens = remaining.split()

            for token in tokens:
                token = token.strip()
                if not token:
                    continue

                # Check if it's a predecessor (pure digit)
                if token.isdigit():
                    predecessors.append(token)
                    if self.debug:
                        print(f"  Predecessor: {token}")
                # Check if it's a valid resource name
                elif is_valid_resource_name(token):
                    # Split by comma if contains comma
                    if ',' in token:
                        sub_resources = [r.strip()
                                         for r in token.split(',') if r.strip()]
                        for sub_r in sub_resources:
                            if is_valid_resource_name(sub_r):
                                resources.append(sub_r)
                                if self.debug:
                                    print(
                                        f"  Resource (from comma-sep): {sub_r}")
                    else:
                        resources.append(token)
                        if self.debug:
                            print(f"  Resource: {token}")

        # Deduplicate resources while preserving order
        unique_resources = []
        seen = set()
        for r in resources:
            r_clean = r.strip()
            if r_clean and r_clean.lower() not in seen:
                unique_resources.append(r_clean)
                seen.add(r_clean.lower())

        task = {
            "id": task_id,
            "task_name": task_name,
            "duration": duration,
            "start_date": start_date,
            "finish_date": finish_date,
            "predecessors": predecessors,
            "resources": unique_resources
        }

        return self._normalize_task(task)

    def _normalize_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize duration
        dur = str(task.get("duration", "")).strip()
        value = 0.0
        unit = "days"
        raw = dur
        m = re.search(r'([\d.]+)', dur)
        if m:
            try:
                value = float(m.group(1))
            except:
                value = 0.0

        # Normalize dates
        sd_raw = str(task.get("start_date", "")).strip()
        fd_raw = str(task.get("finish_date", "")).strip()
        sd = try_parse_date(sd_raw) or sd_raw or ""
        fd = try_parse_date(fd_raw) or fd_raw or ""

        preds = task.get("predecessors", [])
        resources = task.get("resources", [])

        return {
            "id": str(task.get("id", "")).strip(),
            "task_name": str(task.get("task_name", "")).strip(),
            "duration": {
                "value": value,
                "unit": unit,
                "raw": raw
            },
            "start_date": sd,
            "finish_date": fd,
            "predecessors": preds,
            "resources": resources
        }

    def save_json(self, data: Dict[str, Any], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved {len(data.get('tasks', []))} tasks to {output_path}")
        print(f"📊 Project: {data['project']['name']}")
        print(f"📋 Total tasks (expanded): {data['project']['total_tasks']}")
        print(
            f"🔍 Total tasks scanned: {data['project']['total_tasks_scanned']}")
        print(
            f"📦 Original tasks with resources: {data['project']['original_tasks_with_resources']}")

        # Show resources found
        resources = set()
        for task in data['tasks']:
            if task.get('resource'):
                resources.add(task['resource'])

        if resources:
            print(f"\n👥 Resources found ({len(resources)}):")
            for r in sorted(resources):
                count = sum(1 for t in data['tasks'] if t.get('resource') == r)
                print(f"   - {r}: {count} tasks")

        # Show sample of extracted tasks
        if data.get('tasks'):
            print(f"\n📝 Sample tasks extracted:")
            for task in data['tasks'][:5]:
                print(f"   [{task['id']}] {task['task_name']}")
                print(f"      → Resource: {task.get('resource', 'N/A')}")

    def save_txt(self, data: Dict[str, Any], output_path: str):
        lines = []
        for t in data["tasks"]:
            lines.append(
                f"[{t['id']}] {t['task_name']} -> {t.get('resource', '')}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ Saved TXT to {output_path}")

    def save_md(self, data: Dict[str, Any], output_path: str):
        md = [f"# Project: {data['project']['name']}\n"]
        for t in data["tasks"]:
            md.append(
                f"### Task {t['id']}\n- **Name:** {t['task_name']}\n- **Resource:** {t.get('resource','')}\n- **Start:** {t['start_date']}\n- **Finish:** {t['finish_date']}\n")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        print(f"✅ Saved Markdown to {output_path}")

# ------------------------------
# CLI / Runner
# ------------------------------


def main():
    OUTPUT_DIR = "../extract"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Project Plan PDF extractor - Splits tasks by resource")
    parser.add_argument(
        "pdf", nargs="?", help="Path to PDF file (or omit and use --dir)")
    parser.add_argument(
        "--dir", help="Directory to scan for PDF files", default=None)
    parser.add_argument(
        "--out", help="Output JSON path (if single file)", default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug prints")
    args = parser.parse_args()

    extractor = ProjectPlanExtractor(debug=args.debug)

    if args.dir:
        pdfs = [os.path.join(args.dir, p) for p in os.listdir(
            args.dir) if p.lower().endswith(".pdf")]
        if not pdfs:
            print("No PDFs found in directory.")
            return

        processed = 0
        skipped = 0

        for pdf in pdfs:
            try:
                print(f"\n{'='*60}")
                print(f"Processing: {os.path.basename(pdf)}")
                print('='*60)

                data = extractor.extract_from_pdf(pdf)

                if data is None:
                    print(f"⏭️  Skipped: No tasks with resources")
                    skipped += 1
                    continue

                json_path = os.path.join(OUTPUT_DIR, os.path.splitext(
                    os.path.basename(pdf))[0] + ".json")
                txt_path = os.path.join(OUTPUT_DIR, os.path.splitext(
                    os.path.basename(pdf))[0] + ".txt")
                md_path = os.path.join(OUTPUT_DIR, os.path.splitext(
                    os.path.basename(pdf))[0] + ".md")

                extractor.save_json(data, json_path)
                extractor.save_txt(data, txt_path)
                extractor.save_md(data, md_path)
                processed += 1

            except Exception as e:
                print(f"❌ Error processing {pdf}: {e}")
                skipped += 1
                if args.debug:
                    import traceback
                    traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {processed} files")
        print(f"⏭️  Skipped (no resources): {skipped} files")
        print(f"📁 Output directory: {OUTPUT_DIR}")

    else:
        if not args.pdf:
            print("No PDF provided. Use positional argument or --dir.")
            return

        pdf = args.pdf
        json_path = os.path.join(OUTPUT_DIR, os.path.splitext(
            os.path.basename(pdf))[0] + ".json")
        txt_path = os.path.join(OUTPUT_DIR, os.path.splitext(
            os.path.basename(pdf))[0] + ".txt")
        md_path = os.path.join(OUTPUT_DIR, os.path.splitext(
            os.path.basename(pdf))[0] + ".md")

        try:
            print(f"\n{'='*60}")
            print(f"Processing: {pdf}")
            print('='*60)

            data = extractor.extract_from_pdf(pdf)

            if data is None:
                print(f"\n⏭️  No output files created: PDF has no tasks with resources")
                return

            extractor.save_json(data, json_path)
            extractor.save_txt(data, txt_path)
            extractor.save_md(data, md_path)

        except Exception as e:
            print(f"❌ Error processing {pdf}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
