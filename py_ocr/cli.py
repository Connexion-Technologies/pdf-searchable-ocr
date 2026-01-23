#!/usr/bin/env python3
"""
Command-line interface for pdf-searchable-ocr
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
from py_ocr import OCRProcessor


def process_single_file(
    ocr: OCRProcessor,
    input_path: Path,
    output_prefix: str,
    args: argparse.Namespace
) -> Dict[str, any]:
    """
    Process a single file and return result information.

    Args:
        ocr: OCRProcessor instance
        input_path: Path to input file
        output_prefix: Output prefix for generated files
        args: Command line arguments

    Returns:
        Dict with processing results and generated files
    """
    result = {
        'input_file': str(input_path),
        'success': False,
        'is_pdf': False,
        'output_files': [],
        'page_count': 0
    }

    is_pdf = input_path.suffix.lower() == '.pdf'
    result['is_pdf'] = is_pdf

    if is_pdf:
        ocr_results = ocr.process_pdf(str(input_path), output_prefix, args.boxes, args.dpi, args.workers)
        if ocr_results:
            result['success'] = True
            result['page_count'] = len(ocr_results)
            if not args.no_pdf:
                pdf_path = f"{output_prefix}_searchable.pdf"
                # Apply Ghostscript compression if requested
                if args.compress:
                    ocr.compress_pdf_ghostscript(pdf_path, preset=args.compress)
                result['output_files'].append(pdf_path)
            if args.boxes:
                for i in range(len(ocr_results)):
                    if ocr_results[i]:  # Only add if page has results
                        result['output_files'].append(f"{output_prefix}_page_{i+1}_with_boxes.jpg")
    else:
        ocr_results = [ocr.process_image(str(input_path))]
        if ocr_results and ocr_results[0]:
            result['success'] = True
            result['page_count'] = 1

            if not args.no_pdf:
                pdf_path = f"{output_prefix}_searchable.pdf"
                ocr.create_searchable_pdf([ocr_results[0]['image_path']], ocr_results, pdf_path)
                # Apply Ghostscript compression if requested
                if args.compress:
                    ocr.compress_pdf_ghostscript(pdf_path, preset=args.compress)
                result['output_files'].append(pdf_path)

            if args.boxes:
                box_path = f"{output_prefix}_with_boxes.jpg"
                ocr.draw_bounding_boxes(ocr_results[0]['image_path'], ocr_results[0], box_path)
                result['output_files'].append(box_path)

    return result


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="OCR processing with searchable PDF generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pdf-searchable-ocr document.pdf                       # OCR a PDF
  pdf-searchable-ocr file1.pdf file2.pdf                # Process multiple PDFs
  pdf-searchable-ocr *.pdf                              # Process all PDFs in directory
  pdf-searchable-ocr image.jpg --output invoice         # Custom output prefix
  pdf-searchable-ocr image.jpg --lang ch                # Chinese OCR
  pdf-searchable-ocr image.jpg --gpu                    # Use GPU acceleration
  pdf-searchable-ocr document.pdf --dpi 600             # Process PDF at 600 DPI
  pdf-searchable-ocr image.jpg --no-pdf --boxes         # Only generate bounding box image
  pdf-searchable-ocr image.jpg --boxes                  # Generate PDF and bounding box image
  pdf-searchable-ocr document.pdf --quality 50          # Custom JPEG quality (1-100)
  pdf-searchable-ocr document.pdf --compress ebook      # Ghostscript compression (smallest files)
        """
    )

    parser.add_argument("input_files", nargs='+', help="Path(s) to input image or PDF file(s)")
    parser.add_argument("--output", "-o", default=None, help="Output prefix for generated files (default: derived from input filename)")
    parser.add_argument("--lang", "-l", default="en", help="Language for OCR recognition (default: en)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for PDF processing (default: 300)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (requires CUDA)")
    parser.add_argument("--no-pdf", action="store_true", help="Skip searchable PDF generation")
    parser.add_argument("--boxes", "-b", action="store_true", help="Generate bounding box visualization images")
    parser.add_argument("--quality", "-Q", type=int, default=95, help="JPEG quality for PDF images (1-100, default: 95)")
    parser.add_argument("--compress", "-c", choices=["screen", "ebook", "printer", "prepress"],
                        help="Ghostscript compression preset (screen=72dpi, ebook=150dpi, printer/prepress=300dpi)")
    parser.add_argument("--workers", "-j", type=int, default=1, help="Number of parallel workers for OCR (default: 1)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    parser.add_argument("--version", action="version", version="pdf-searchable-ocr 0.1.1")
    
    args = parser.parse_args()

    # Validate input files
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.pdf']
    input_paths = []
    for file_str in args.input_files:
        input_path = Path(file_str)
        if not input_path.exists():
            print(f"❌ Error: Input file not found: {file_str}")
            return 1
        if input_path.suffix.lower() not in supported_formats:
            print(f"❌ Error: Unsupported file format: {input_path.suffix} for file {file_str}")
            return 1
        input_paths.append(input_path)

    try:
        # Initialize OCR processor once for all files
        ocr = OCRProcessor(lang=args.lang, use_gpu=args.gpu, verbose=not args.quiet)

        # Apply image quality setting
        ocr.set_image_quality(args.quality)

        # Process each file
        total_files = len(input_paths)
        results = []

        for idx, input_path in enumerate(input_paths, 1):
            # Determine output prefix
            if args.output:
                # If user specified output prefix, use it
                # For multiple files, append index if more than one file
                if total_files > 1:
                    output_prefix = f"{args.output}_{input_path.stem}"
                else:
                    output_prefix = args.output
            else:
                # Default: use input filename stem
                output_prefix = input_path.stem

            # Show progress for multiple files
            if total_files > 1 and not args.quiet:
                print(f"\n{'='*60}")
                print(f"Processing file {idx}/{total_files}: {input_path.name}")
                print(f"{'='*60}")

            # Process the file
            result = process_single_file(ocr, input_path, output_prefix, args)
            results.append(result)

            if not result['success']:
                print(f"❌ Error: OCR processing failed for {input_path.name}")

        # Display summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print("🎉 Processing Summary")
            print(f"{'='*60}")

            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]

            print(f"✅ Successfully processed: {len(successful)}/{total_files} files")
            if failed:
                print(f"❌ Failed: {len(failed)} files")
                for r in failed:
                    print(f"   - {Path(r['input_file']).name}")

            if successful:
                print("\n📁 Generated files:")
                for r in successful:
                    print(f"\n  {Path(r['input_file']).name} ({r['page_count']} page{'s' if r['page_count'] > 1 else ''}):")
                    for output_file in r['output_files']:
                        print(f"    📄 {output_file}")

        # Return error code if any file failed
        return 1 if any(not r['success'] for r in results) else 0

    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())