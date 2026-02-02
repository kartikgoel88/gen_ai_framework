"""Document Processing Example.

This example demonstrates document extraction from various formats:
- PDF
- DOCX
- TXT
- Excel
- OCR for images
"""

from pathlib import Path
from src.framework.config import get_settings
from src.framework.api.deps import get_document_processor, get_ocr_processor


def main():
    # Get settings and components
    settings = get_settings()
    doc_processor = get_document_processor(settings)
    ocr_processor = get_ocr_processor()
    
    print("=" * 60)
    print("Document Processing Examples")
    print("=" * 60 + "\n")
    
    # Create a sample text file
    sample_dir = Path(settings.UPLOAD_DIR) / "examples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Text file
    print("1. Text File Extraction:")
    print("-" * 60)
    txt_file = sample_dir / "sample.txt"
    txt_file.write_text("This is a sample text file.\nIt contains multiple lines.\n")
    result = doc_processor.extract(str(txt_file))
    print(f"Extracted: {result.text}\n")
    
    # 2. Document processor info
    print("2. Document Processor Info:")
    print("-" * 60)
    print(f"Upload Directory: {doc_processor.upload_dir}")
    print(f"Supported Types: PDF, DOCX, TXT, Excel\n")
    
    # 3. OCR Example (if image available)
    print("3. OCR Processing:")
    print("-" * 60)
    print("Note: OCR requires an image file (PNG, JPG)")
    print("Example usage:")
    print("  from pathlib import Path")
    print("  image_bytes = Path('image.png').read_bytes()")
    print("  result = ocr_processor.extract_from_bytes(image_bytes)")
    print("  print(result.text)\n")
    
    # 4. File upload simulation
    print("4. File Upload Simulation:")
    print("-" * 60)
    print("In a real application, files come from:")
    print("- FastAPI file uploads")
    print("- Streamlit file uploader")
    print("- CLI arguments")
    print("\nExample:")
    print("  uploaded_file = st.file_uploader('Upload document')")
    print("  if uploaded_file:")
    print("      path = save_uploaded_file(uploaded_file)")
    print("      result = doc_processor.extract(path)")


if __name__ == "__main__":
    main()
