# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
from PIL import Image
import io
from datetime import datetime
import json
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from docling.document_converter import DocumentConverter
print("📚 Importing libraries...")
print("=" * 50)


print("✅ Libraries imported!\n")

# ============================================
# STEP 2: INITIALIZE OCR ENGINE
# ============================================
print("🚀 Initializing RapidOCR engine...")
print("=" * 50)

# Initialize RapidOCR (akan auto-download model saat pertama kali)
ocr_engine = RapidOCR()

print("✅ RapidOCR engine ready!\n")

# ============================================
# STEP 3: CONFIGURE DOCLING
# ============================================
print("⚙️ Configuring Docling...")
print("=" * 50)

# Initialize Docling dengan konfigurasi default
# Tidak perlu set pipeline_options manual karena bisa cause error
doc_converter = DocumentConverter()

print("✅ Docling configured!\n")

# ============================================
# STEP 4: DEFINE PROCESSING FUNCTIONS
# ============================================


def process_with_rapidocr(image_path):
    """
    Process image dengan RapidOCR

    Args:
        image_path: Path ke file gambar

    Returns:
        tuple: (text, result_details)
    """
    print(f"🔍 Running RapidOCR on: {image_path}")

    try:
        result, elapse = ocr_engine(str(image_path))

        if result:
            # Extract text dari hasil OCR
            text_lines = [line[1] for line in result]
            text = "\n".join(text_lines)

            print(f"✅ OCR completed in {elapse:.2f}s")
            print(f"📝 Detected {len(text_lines)} lines")

            # Show confidence scores
            confidences = [line[2] for line in result]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"🎯 Average confidence: {avg_confidence:.2%}\n")

            return text, result
        else:
            print("⚠️ No text detected\n")
            return "", None
    except Exception as e:
        print(f"❌ RapidOCR Error: {e}\n")
        return "", None


def process_document(file_path, use_rapidocr=True):
    """
    Process dokumen dengan Docling + RapidOCR

    Args:
        file_path: Path ke file dokumen
        use_rapidocr: Boolean, apakah menggunakan RapidOCR untuk gambar

    Returns:
        tuple: (result_data dict, doc_result object)
    """
    file_path = Path(file_path)
    print(f"\n{'='*50}")
    print(f"📄 Processing: {file_path.name}")
    print(f"{'='*50}\n")

    result_data = {
        "filename": file_path.name,
        "processed_at": datetime.now().isoformat(),
        "file_type": file_path.suffix,
    }

    # Process dengan Docling
    print("🔄 Converting document with Docling...")
    try:
        doc_result = doc_converter.convert(file_path)

        # Get markdown content
        markdown_content = doc_result.document.export_to_markdown()
        result_data["docling_text"] = markdown_content
        print(f"✅ Docling extraction: {len(markdown_content)} characters\n")
    except Exception as e:
        print(f"⚠️ Docling extraction warning: {e}")
        markdown_content = ""
        result_data["docling_text"] = ""
        doc_result = None

    # Process dengan RapidOCR jika file adalah gambar
    ocr_text = ""
    ocr_details = None

    if use_rapidocr and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        print("🖼️ Detected image file, running RapidOCR...\n")
        ocr_text, ocr_details = process_with_rapidocr(str(file_path))
        result_data["rapidocr_text"] = ocr_text
        if ocr_details:
            # Simplify details for JSON serialization
            result_data["rapidocr_line_count"] = len(ocr_details)

    # Jika PDF, coba juga dengan RapidOCR (untuk scanned PDF)
    elif use_rapidocr and file_path.suffix.lower() == '.pdf' and len(markdown_content) < 100:
        print("📄 PDF with little text detected, trying RapidOCR on first page...\n")
        # Convert first page of PDF to image and OCR
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap()
                img_path = f"{file_path.stem}_temp.png"
                pix.save(img_path)
                ocr_text, ocr_details = process_with_rapidocr(img_path)
                result_data["rapidocr_text"] = ocr_text
                # Clean up temp file
                Path(img_path).unlink(missing_ok=True)
            doc.close()
        except ImportError:
            print("⚠️ PyMuPDF not installed, skipping PDF image extraction")
        except Exception as e:
            print(f"⚠️ PDF image extraction error: {e}")

    # Combine hasil
    full_text = markdown_content
    if ocr_text:
        if full_text:
            full_text += "\n\n" + "="*50 + "\n"
            full_text += "RapidOCR RESULTS\n"
            full_text += "="*50 + "\n\n"
        full_text += ocr_text

    result_data["combined_text"] = full_text
    result_data["text_length"] = len(full_text)

    return result_data, doc_result


def display_results(result_data):
    """Display hasil processing dengan format yang rapi"""
    print(f"\n{'='*50}")
    print("📊 PROCESSING RESULTS")
    print(f"{'='*50}\n")

    print(f"📁 Filename: {result_data['filename']}")
    print(f"📅 Processed: {result_data['processed_at']}")
    print(f"📝 Total text length: {result_data['text_length']} characters")

    if 'rapidocr_line_count' in result_data:
        print(f"🔍 OCR detected: {result_data['rapidocr_line_count']} lines")

    print(f"\n{'-'*50}")
    print("📄 EXTRACTED TEXT:")
    print(f"{'-'*50}\n")

    if result_data['combined_text']:
        preview_text = result_data['combined_text'][:1500]
        print(preview_text)

        if len(result_data['combined_text']) > 1500:
            remaining = len(result_data['combined_text']) - 1500
            print(f"\n... (truncated, {remaining} more characters)")
    else:
        print("⚠️ No text extracted from document")

    print(f"\n{'-'*50}\n")


def save_results(result_data, doc_result, output_prefix="output"):
    """Save hasil ke file"""
    print(f"\n💾 Saving results...")

    saved_files = {}

    # Save as JSON
    json_path = f"{output_prefix}_extracted.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        # Remove non-serializable items
        save_data = {k: v for k, v in result_data.items()
                     if k not in ['rapidocr_details']}
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved JSON: {json_path}")
    saved_files['json'] = json_path

    # Save as Markdown
    md_path = f"{output_prefix}_extracted.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {result_data['filename']}\n\n")
        f.write(f"**Processed:** {result_data['processed_at']}\n\n")
        f.write(f"---\n\n")
        f.write(result_data['combined_text'])
    print(f"✅ Saved Markdown: {md_path}")
    saved_files['markdown'] = md_path

    # Save as TXT
    txt_path = f"{output_prefix}_extracted.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result_data['combined_text'])
    print(f"✅ Saved TXT: {txt_path}")
    saved_files['txt'] = txt_path

    # Save as PDF (jika ada doc_result dari Docling)
    if doc_result:
        try:
            pdf_path = f"{output_prefix}_processed.pdf"
            doc_result.document.save_as_pdf(pdf_path)
            print(f"✅ Saved PDF: {pdf_path}")
            saved_files['pdf'] = pdf_path
        except Exception as e:
            print(f"⚠️ Could not save PDF: {e}")

    return saved_files


# ============================================
# STEP 5: UPLOAD FILE DARI KOMPUTER
# ============================================
print(f"\n{'='*50}")
print("📤 UPLOAD YOUR DOCUMENT")
print(f"{'='*50}\n")
print("Supported formats:")
print("  - PDF (.pdf)")
print("  - Images (.png, .jpg, .jpeg, .tiff, .bmp, .gif)")
print("  - Word documents (.docx)")
print("  - PowerPoint (.pptx)")
print("  - Excel (.xlsx)")
print("\nClick 'Choose Files' button below to upload...\n")

# ============================================
# STEP 6: PROCESS FILE FROM PATH
# ============================================

# Ganti path file kamu di sini 👇
file_path = "../DocTimeline/SCB - DHN Application - Project Plan v.01.1.pdf"
# file_path = "C:\\Users\\dzikri\\Documents\\file.pdf"  # contoh path di Windows

# Folder output
output_dir = Path("../extract")
output_dir.mkdir(parents=True, exist_ok=True)

if Path(file_path).exists():
    print(f"\n✅ File ditemukan: {file_path}")
    print(f"📦 File size: {Path(file_path).stat().st_size / 1024:.2f} KB\n")

    try:
        # Proses dokumen
        result_data, doc_result = process_document(
            file_path, use_rapidocr=True)

        # Tampilkan hasil
        display_results(result_data)

        # Tentukan prefix hasil (disimpan di ../extract)
        output_prefix = output_dir / Path(file_path).stem

        # Simpan hasil
        output_files = save_results(result_data, doc_result,
                                    output_prefix=output_prefix)

        # Info hasil
        print(f"\n{'='*50}")
        print("💾 HASIL TERSIMPAN")
        print(f"{'='*50}\n")

        for file_type, file_path in output_files.items():
            print(f"✅ {file_type.upper()} disimpan di: {file_path}")

        print(f"\n{'='*50}")
        print("✅ SELESAI SEMUA!")
        print(f"{'='*50}\n")

        # Statistik singkat
        print("📊 Statistik Cepat:")
        print(f"   • Karakter diekstrak: {result_data['text_length']}")
        print(f"   • File dihasilkan: {len(output_files)}")
        print(f"   • Waktu proses: {datetime.now().isoformat()}")

    except Exception as e:
        print(f"\n❌ TERJADI ERROR SAAT PROSES FILE:")
        print(f"   {str(e)}")
        print(f"\nCoba langkah berikut:")
        print("   1. Pastikan format file didukung")
        print("   2. Coba file lain")
        print("   3. Laporkan error di atas jika masih muncul")

else:
    print(f"⚠️ File tidak ditemukan di path: {file_path}")
    print("   Pastikan path-nya benar dan file tersedia.")