#!/usr/bin/env python3
"""
Command-line interface for pdf-searchable-ocr
"""

import argparse
import sys
from pathlib import Path
from py_ocr import OCRProcessor


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="OCR processing with searchable PDF generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf-searchable-ocr document.pdf                       # OCR a PDF
  pdf-searchable-ocr image.jpg --output invoice         # Custom output prefix
  pdf-searchable-ocr image.jpg --lang ch                # Chinese OCR
  pdf-searchable-ocr image.jpg --gpu                    # Use GPU acceleration
  pdf-searchable-ocr document.pdf --dpi 600             # Process PDF at 600 DPI
  pdf-searchable-ocr image.jpg --no-pdf                 # Only generate bounding box image
  pdf-searchable-ocr image.jpg --no-boxes               # Only generate searchable PDF
        """
    )
    
    parser.add_argument("input_file", help="Path to the input image or PDF file")
    parser.add_argument("--output", "-o", default="output", help="Output prefix for generated files (default: output)")
    parser.add_argument("--lang", "-l", default="en", help="Language for OCR recognition (default: en)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for PDF processing (default: 300)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (requires CUDA)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip searchable PDF generation")
    parser.add_argument("--no-boxes", action="store_true", help="Skip bounding box image generation")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    parser.add_argument("--version", action="version", version="pdf-searchable-ocr 0.1.1")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input_file}")
        return 1

    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
    if input_path.suffix.lower() not in supported_formats:
        print(f"❌ Error: Unsupported file format: {input_path.suffix}")
        return 1

    try:
        ocr = OCRProcessor(lang=args.lang, use_gpu=args.gpu, verbose=not args.quiet)
        is_pdf = input_path.suffix.lower() == '.pdf'

        if is_pdf:
            ocr_results = ocr.process_pdf(str(input_path), args.output, not args.no_boxes, args.dpi)
        else:
            ocr_results = [ocr.process_image(str(input_path))]
            if not ocr_results or not any(ocr_results):
                print("❌ Error: OCR processing failed for image")
                return 1

            if not args.no_pdf:
                pdf_path = f"{args.output}_searchable.pdf"
                ocr.create_searchable_pdf([ocr_results[0]['image_path']], ocr_results, pdf_path)

            if not args.no_boxes:
                box_path = f"{args.output}_with_boxes.jpg"
                ocr.draw_bounding_boxes(ocr_results[0]['image_path'], ocr_results[0], box_path)

        if not ocr_results:
            print("❌ Error: OCR processing failed")
            return 1

        if not args.quiet:
            print("\n🎉 Processing completed successfully!")
            print("📁 Generated files:")
            if not args.no_pdf:
                print(f"   📄 {args.output}_searchable.pdf - Searchable PDF")
            if not args.no_boxes:
                if is_pdf:
                    print("   🎨 Bounding box images:")
                    for i in range(len(ocr_results)):
                        print(f"      - {args.output}_page_{i+1}_with_boxes.jpg")
                else:
                    print(f"   🎨 {args.output}_with_boxes.jpg - Image with bounding boxes")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())