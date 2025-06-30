#!/usr/bin/env python3
"""
Extract text from Sakana AI automated research paper
"""

import PyPDF2
import sys
import os

def extract_pdf_text(pdf_path, output_path):
    """Extract text from PDF and save to file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"Extracting text from {len(pdf_reader.pages)} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n--- PAGE {page_num + 1} ---\n"
                    text += page_text
                except Exception as e:
                    print(f"Error extracting page {page_num + 1}: {e}")
                    continue
            
            # Save extracted text
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            
            print(f"Text extracted successfully to: {output_path}")
            print(f"Total characters extracted: {len(text)}")
            
            return text
            
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "/Users/ryneschultz/Documents/GitHub/PRSM/docs/2408.06292v3.pdf"
    output_path = "/Users/ryneschultz/Documents/GitHub/PRSM/docs/2408.06292v3_extracted_text.txt"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    text = extract_pdf_text(pdf_path, output_path)
    
    if text:
        print("\nFirst 500 characters of extracted text:")
        print("-" * 50)
        print(text[:500])
        print("-" * 50)
    else:
        print("Failed to extract text from PDF")