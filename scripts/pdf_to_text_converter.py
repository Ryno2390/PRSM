#!/usr/bin/env python3
"""
PDF to Text Converter using PyPDF2
Converts the 5299044.pdf file to readable text format
"""

import PyPDF2
import sys
import os

def extract_text_from_pdf(pdf_path, output_path=None):
    """
    Extract text from PDF file using PyPDF2
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save extracted text. If None, returns text.
    
    Returns:
        str: Extracted text content
    """
    try:
        with open(pdf_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get number of pages
            num_pages = len(pdf_reader.pages)
            print(f"PDF contains {num_pages} pages")
            
            # Extract text from all pages
            extracted_text = ""
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                extracted_text += f"\n--- PAGE {page_num + 1} ---\n"
                extracted_text += page_text
                extracted_text += "\n"
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(extracted_text)
                print(f"Text extracted and saved to: {output_path}")
            
            return extracted_text
            
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

if __name__ == "__main__":
    # Path to the PDF file
    pdf_file = "/Users/ryneschultz/Documents/GitHub/PRSM/docs/5299044.pdf"
    output_file = "/Users/ryneschultz/Documents/GitHub/PRSM/5299044_extracted_text.txt"
    
    # Check if PDF file exists
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at {pdf_file}")
        sys.exit(1)
    
    print(f"Extracting text from: {pdf_file}")
    text_content = extract_text_from_pdf(pdf_file, output_file)
    
    if text_content:
        print("Text extraction completed successfully!")
        print(f"Extracted text length: {len(text_content)} characters")
        
        # Show first 500 characters as preview
        print("\n=== PREVIEW (First 500 characters) ===")
        print(text_content[:500])
        print("...")
    else:
        print("Failed to extract text from PDF")
        sys.exit(1)