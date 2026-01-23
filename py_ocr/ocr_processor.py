#!/usr/bin/env python3
"""
OCRProcessor: A class-based OCR processor with searchable PDF generation
"""

import os
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Optional, Tuple, Dict, List, Any
import io
import cv2
from PIL import Image
from paddleocr import PaddleOCR
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from pdf2image import convert_from_path


class OCRProcessor:
    """
    A class for processing images with OCR and generating searchable PDFs.
    
    This class provides functionality to:
    - Perform OCR on images using PaddleOCR
    - Generate searchable PDFs with invisible text layers
    - Draw bounding boxes on images for visualization
    - Handle various image formats
    
    Attributes:
        ocr_engine (PaddleOCR): The PaddleOCR instance
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(self,
                 lang: str = 'en',
                 use_gpu: bool = False,
                 verbose: bool = True,
                 **kwargs):
        """
        Initialize the OCR processor.

        Args:
            lang (str): Language for OCR recognition (default: 'en')
            use_gpu (bool): Whether to use GPU acceleration (default: False)
            verbose (bool): Whether to print verbose output (default: True)
            **kwargs: Additional arguments passed to PaddleOCR
        """
        self.verbose = verbose
        self.font_path = None
        self.image_quality = 85  # JPEG quality for PDF embedding (1-100)
        
        # Initialize PaddleOCR with safe settings
        ocr_kwargs = {
            'lang': lang,
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': True,
        }
        
        # Add GPU support if available and requested
        if use_gpu:
            ocr_kwargs['use_gpu'] = True
        
        # Add additional kwargs safely
        for key, value in kwargs.items():
            ocr_kwargs[key] = value
        
        if self.verbose:
            print(f"🔧 Initializing OCR engine with language: {lang}")
        
        try:
            self.ocr_engine = PaddleOCR(**ocr_kwargs)
        except ValueError as e:
            if 'use_gpu' in str(e) and use_gpu:
                # Fallback to CPU if GPU not supported
                if self.verbose:
                    print("⚠️  GPU not supported, falling back to CPU")
                ocr_kwargs.pop('use_gpu', None)
                self.ocr_engine = PaddleOCR(**ocr_kwargs)
            else:
                raise
        
        if self.verbose:
            print("✅ OCR engine initialized successfully")

    def set_image_quality(self, quality: int) -> None:
        """
        Set the JPEG quality for PDF image embedding.

        Args:
            quality (int): JPEG quality (1-100). Higher values mean better quality but larger files.
                          Recommended: 85 for normal, 65 for reduced size.
        """
        self.image_quality = max(1, min(100, quality))
        if self.verbose:
            print(f"📊 Image quality set to: {self.image_quality}")

    def _find_font_path(self) -> Optional[str]:
        """
        Find a suitable CJK font from common system paths.

        Returns:
            str: Path to a valid font file, or None if not found
        """
        # Common font paths across different systems
        # Prioritize TTF files as TTC files may not work with reportlab
        font_paths = [
            # Nanum fonts (TTF, widely compatible)
            '/usr/share/fonts/TTF/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/nanum/NanumGothic.ttf',
            # Noto Sans KR (single language, better compatibility)
            '/usr/share/fonts/noto/NotoSansKR-Regular.ttf',
            '/usr/share/fonts/TTF/NotoSansKR-Regular.ttf',
            '/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf',
            # Noto Sans JP
            '/usr/share/fonts/noto/NotoSansJP-Regular.ttf',
            '/usr/share/fonts/TTF/NotoSansJP-Regular.ttf',
            # Noto Sans SC (Simplified Chinese)
            '/usr/share/fonts/noto/NotoSansSC-Regular.ttf',
            '/usr/share/fonts/TTF/NotoSansSC-Regular.ttf',
            # DejaVu (has some CJK support)
            '/usr/share/fonts/TTF/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            # macOS
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/Library/Fonts/Arial Unicode.ttf',
            # Windows
            'C:\\Windows\\Fonts\\msyh.ttc',  # Microsoft YaHei
            'C:\\Windows\\Fonts\\malgun.ttf',  # Malgun Gothic
            'C:\\Windows\\Fonts\\arial.ttf',
        ]

        for path in font_paths:
            if os.path.exists(path):
                if self.verbose:
                    print(f"✅ Found font: {path}")
                return path

        if self.verbose:
            print("⚠️ No CJK font found in common paths")
        return None

    def download_sample_image(self, url: str = None, filename: str = "sample_image.jpg") -> Optional[str]:
        """
        Download a sample image for testing.
        
        Args:
            url (str, optional): URL to download image from
            filename (str): Local filename to save the image
            
        Returns:
            str: Path to the downloaded image, or None if failed
        """
        if url is None:
            url = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/doc/imgs/11.jpg"
        
        if not os.path.exists(filename):
            if self.verbose:
                print(f"📥 Downloading sample image from {url}")
            try:
                urllib.request.urlretrieve(url, filename)
                if self.verbose:
                    print(f"✅ Sample image saved as {filename}")
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to download image: {e}")
                return None
        else:
            if self.verbose:
                print(f"📁 Using existing sample image: {filename}")
        
        return filename
    
    def process_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Perform OCR on an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: OCR results containing texts, scores, and bounding boxes
            None: If OCR failed
        """
        if not os.path.exists(image_path):
            if self.verbose:
                print(f"❌ Image file not found: {image_path}")
            return None
        
        try:
            if self.verbose:
                print(f"🔍 Processing image: {image_path}")
            
            # Perform OCR
            result = self.ocr_engine.ocr(image_path)
            
            if not result or len(result) == 0:
                if self.verbose:
                    print("❌ No OCR results")
                return None
            
            ocr_result = result[0]
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [])
            rec_boxes = ocr_result.get('rec_boxes', [])
            
            if not rec_texts:
                if self.verbose:
                    print("❌ No text detected")
                return None
            
            if self.verbose:
                avg_confidence = sum(rec_scores) / len(rec_scores) if rec_scores else 0
                print(f"📊 Text blocks detected: {len(rec_texts)} | Average confidence: {avg_confidence:.3f}")
            
            return {
                'rec_texts': rec_texts,
                'rec_scores': rec_scores,
                'rec_boxes': rec_boxes,
                'image_path': image_path
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error during OCR processing: {e}")
            return None
    
    def create_searchable_pdf(self,
                            image_paths: List[str],
                            ocr_results: List[Dict[str, Any]],
                            output_pdf: str = "searchable_output.pdf") -> Optional[str]:
        """
        Create a searchable PDF with invisible text layers from multiple pages.

        Args:
            image_paths (List[str]): List of paths to the source images
            ocr_results (List[Dict[str, Any]]): List of OCR results from process_image()
            output_pdf (str): Output PDF filename

        Returns:
            str: Path to the created PDF, or None if failed
        """
        try:
            if not image_paths or not ocr_results:
                if self.verbose:
                    print("❌ No images or OCR results to create PDF")
                return None

            # Find and register a suitable font
            if not self.font_path:
                self.font_path = self._find_font_path()

            font_name = "Helvetica"  # Default fallback font
            if self.font_path:
                try:
                    pdfmetrics.registerFont(TTFont('CJKFont', self.font_path))
                    font_name = 'CJKFont'
                    if self.verbose:
                        print(f"📝 Using font: {self.font_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Failed to register font, using Helvetica: {e}")
            else:
                if self.verbose:
                    print("⚠️ No CJK font found, using Helvetica (may not display some characters correctly)")

            # Get dimensions from the first image
            first_image = cv2.imread(image_paths[0])
            if first_image is None:
                if self.verbose:
                    print(f"❌ Could not read image: {image_paths[0]}")
                return None

            img_height, img_width = first_image.shape[:2]

            # Create PDF with consistent page size
            c = canvas.Canvas(output_pdf, pagesize=(img_width, img_height))

            for i, (image_path, ocr_result) in enumerate(zip(image_paths, ocr_results)):
                if self.verbose:
                    print(f"📄 Adding page {i+1} to PDF...")

                # Compress and add the image as background
                img = Image.open(image_path)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=self.image_quality, optimize=True)
                img_buffer.seek(0)
                c.drawImage(ImageReader(img_buffer), 0, 0, width=img_width, height=img_height)

                # Add invisible text layers
                if ocr_result and ocr_result.get('rec_texts'):
                    for text, score, box in zip(ocr_result['rec_texts'], ocr_result['rec_scores'], ocr_result['rec_boxes']):
                        if text.strip():
                            try:
                                x1, y1, x2, y2 = box[:4]
                                pdf_x = x1
                                pdf_y = img_height - y2
                                text_height = y2 - y1
                                font_size = max(8, min(text_height * 0.8, 48))
                                c.setFillColorRGB(0, 0, 0, alpha=0)
                                c.setFont(font_name, font_size)
                                c.drawString(pdf_x, pdf_y, text)
                            except Exception as e:
                                if self.verbose:
                                    print(f"⚠️ Error adding text '{text}': {e}")
                else:
                    if self.verbose:
                        print(f"⚠️ No text found for page {i+1}")

                c.showPage()

            c.save()

            if self.verbose:
                print(f"✅ Searchable PDF saved as: {output_pdf}")

            return output_pdf

        except Exception as e:
            if self.verbose:
                print(f"❌ Error creating searchable PDF: {e}")
            return None
    
    def draw_bounding_boxes(self, 
                          image_path: str, 
                          ocr_result: Dict[str, Any], 
                          output_image: str = "image_with_boxes.jpg") -> Optional[str]:
        """
        Draw bounding boxes on the image to visualize OCR detection.
        
        Args:
            image_path (str): Path to the source image
            ocr_result (dict): OCR results from process_image()
            output_image (str): Output image filename
            
        Returns:
            str: Path to the image with bounding boxes, or None if failed
        """
        try:
            # Extract OCR data
            rec_texts = ocr_result.get('rec_texts', [])
            rec_scores = ocr_result.get('rec_scores', [])
            rec_boxes = ocr_result.get('rec_boxes', [])
            
            if not rec_texts:
                if self.verbose:
                    print("❌ No text found to draw bounding boxes")
                return None
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose:
                    print(f"❌ Could not read image: {image_path}")
                return None
            
            if self.verbose:
                print(f"🎨 Drawing {len(rec_texts)} bounding boxes on image...")
            
            # Draw bounding boxes and text labels
            for i, (text, score, box) in enumerate(zip(rec_texts, rec_scores, rec_boxes), 1):
                if not text.strip():  # Skip empty text
                    continue
                    
                try:
                    # Extract bounding box coordinates
                    if len(box) >= 4:
                        x1, y1, x2, y2 = map(int, box[:4])
                    else:
                        if self.verbose:
                            print(f"⚠️  Invalid bounding box for text {i}: {box}")
                        continue
                    
                    # Choose color based on confidence
                    if score > 0.8:
                        color = (0, 255, 0)  # Green for high confidence
                    elif score > 0.5:
                        color = (0, 165, 255)  # Orange for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for low confidence
                    
                    # Draw bounding box rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label text
                    label_text = f"{i}: {text[:20]}..." if len(text) > 20 else f"{i}: {text}"
                    confidence_text = f"({score:.2f})"
                    
                    # Draw text labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                    (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
                    
                    # Draw background rectangle for text
                    label_bg_y = max(y1 - label_h - 10, 0)
                    cv2.rectangle(image, (x1, label_bg_y), (x1 + max(label_w, conf_w) + 10, y1), color, -1)
                    
                    # Draw text labels
                    cv2.putText(image, label_text, (x1 + 2, y1 - label_h - 2), font, font_scale, (255, 255, 255), thickness)
                    cv2.putText(image, confidence_text, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Error drawing box {i}: {e}")
                    continue
            
            # Save the image with bounding boxes
            cv2.imwrite(output_image, image)
            
            if self.verbose:
                print(f"✅ Image with bounding boxes saved as: {output_image}")
            
            return output_image
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error drawing bounding boxes: {e}")
            return None
    
    def process_and_generate_all(self, 
                               image_path: str, 
                               output_pdf: str = "searchable_output.pdf",
                               output_prefix: str = "output", bounding_boxes: bool = False) -> Dict[str, Optional[str]]:
        """
        Complete workflow: OCR + Searchable PDF + Bounding Box Image.
        
        Args:
            image_path (str): Path to the input image
            output_prefix (str): Prefix for output files
            
        Returns:
            dict: Paths to generated files
        """
        results = {
            'ocr_result': None,
            'searchable_pdf': None,
            'boxed_image': None
        }
        
        # Step 1: Perform OCR
        ocr_result = self.process_image(image_path)
        if not ocr_result:
            return results
        
        results['ocr_result'] = ocr_result
        
        # Step 2: Create searchable PDF
        pdf_path = output_pdf
        results['searchable_pdf'] = self.create_searchable_pdf(image_path, ocr_result, pdf_path)
        
        # Step 3: Create image with bounding boxes
        if bounding_boxes:
            boxed_image_path = f"{output_prefix}_with_boxes.jpg"
            results['boxed_image'] = self.draw_bounding_boxes(image_path, ocr_result, boxed_image_path)

        return results

    def process_pdf(self, pdf_path: str, output_prefix: str, bounding_boxes: bool, dpi: int = 300) -> Optional[List[Dict[str, Any]]]:
        """
        Perform OCR on a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_prefix (str): Prefix for output files
            bounding_boxes (bool): Whether to generate bounding box images
            dpi (int): Resolution for PDF to image conversion
            
        Returns:
            List[Dict[str, Any]]: A list of OCR results for each page
        """
        if not os.path.exists(pdf_path):
            if self.verbose:
                print(f"❌ PDF file not found: {pdf_path}")
            return None
        
        try:
            if self.verbose:
                print(f"🔄 Converting PDF to images at {dpi} DPI...")
            
            images = convert_from_path(pdf_path, dpi=dpi)
            
            if not images:
                if self.verbose:
                    print("❌ Could not convert PDF to images")
                return None
            
            if self.verbose:
                print(f"🖼️ PDF converted to {len(images)} pages")
            
            all_results = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                image_paths = []
                for i, image in enumerate(images):
                    page_num = i + 1
                    
                    # Save the image to a temporary file
                    temp_image_path = self.save_temp_image(image, temp_dir, page_num)
                    image_paths.append(temp_image_path)
                    
                    if self.verbose:
                        print(f"🔍 Processing page {page_num}...")
                    
                    # Perform OCR on the temporary image
                    ocr_result = self.process_image(temp_image_path)

                    if ocr_result:
                        all_results.append(ocr_result)
                    else:
                        all_results.append(None)  # Preserve alignment with image_paths
                        if self.verbose:
                            print(f"⚠️ No OCR results for page {page_num}")
                
                if not all_results or not any(all_results):
                    if self.verbose:
                        print("❌ No OCR results found in any page")
                    return None
                
                # Generate searchable PDF
                pdf_output_path = f"{output_prefix}_searchable.pdf"
                self.create_searchable_pdf(image_paths, all_results, pdf_output_path)
                
                # Generate bounding box images
                if bounding_boxes:
                    for i, (img_path, res) in enumerate(zip(image_paths, all_results)):
                        page_num = i + 1
                        if res:
                            box_path = f"{output_prefix}_page_{page_num}_with_boxes.jpg"
                            self.draw_bounding_boxes(img_path, res, box_path)
                        else:
                            if self.verbose:
                                print(f"⏭️  Skipping bounding boxes for blank page {page_num}")
            
            return all_results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error processing PDF: {e}")
            return None

    def save_temp_image(self, image: Image.Image, temp_dir: str, page_num: int) -> str:
        """
        Save a PIL image to a temporary directory as JPEG for efficiency.

        Args:
            image (Image.Image): The PIL image to save
            temp_dir (str): The temporary directory path
            page_num (int): The page number, used for the filename

        Returns:
            str: The path to the saved temporary image
        """
        temp_image_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
        # Convert RGBA to RGB if necessary (JPEG doesn't support alpha)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(temp_image_path, "JPEG", quality=self.image_quality, optimize=True)
        return temp_image_path

    def compress_pdf_ghostscript(self, input_pdf: str, output_pdf: str = None,
                                  preset: str = "ebook") -> Optional[str]:
        """
        Compress a PDF file using Ghostscript.

        Args:
            input_pdf (str): Path to the input PDF file
            output_pdf (str, optional): Path for the output PDF. If None, replaces the input.
            preset (str): Compression preset. Options:
                - "screen": 72 dpi, lowest quality, smallest size
                - "ebook": 150 dpi, medium quality (default)
                - "printer": 300 dpi, high quality
                - "prepress": 300 dpi, highest quality

        Returns:
            str: Path to the compressed PDF, or None if failed
        """
        # Check if Ghostscript is available
        gs_cmd = shutil.which("gs")
        if not gs_cmd:
            if self.verbose:
                print("⚠️ Ghostscript (gs) not found. Install with: sudo pacman -S ghostscript")
            return None

        valid_presets = ["screen", "ebook", "printer", "prepress"]
        if preset not in valid_presets:
            if self.verbose:
                print(f"⚠️ Invalid preset '{preset}'. Using 'ebook'. Valid: {valid_presets}")
            preset = "ebook"

        # If no output specified, compress in-place using temp file
        replace_original = output_pdf is None
        if replace_original:
            output_pdf = input_pdf + ".tmp"

        try:
            if self.verbose:
                print(f"🗜️  Compressing PDF with Ghostscript ({preset} preset)...")

            cmd = [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                f"-dPDFSETTINGS=/{preset}",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                f"-sOutputFile={output_pdf}",
                input_pdf
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                if self.verbose:
                    print(f"❌ Ghostscript error: {result.stderr}")
                return None

            # Replace original if needed
            if replace_original:
                os.replace(output_pdf, input_pdf)
                output_pdf = input_pdf

            # Report size reduction
            if self.verbose:
                original_size = os.path.getsize(input_pdf if not replace_original else output_pdf)
                print(f"✅ Compressed PDF saved: {output_pdf} ({original_size / 1024 / 1024:.2f} MB)")

            return output_pdf

        except Exception as e:
            if self.verbose:
                print(f"❌ Error during Ghostscript compression: {e}")
            # Clean up temp file if it exists
            if replace_original and os.path.exists(output_pdf):
                os.remove(output_pdf)
            return None
