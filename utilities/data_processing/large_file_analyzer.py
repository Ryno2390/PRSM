#!/usr/bin/env python3
"""
Large File Analyzer - Safely extract content from large text files
"""

import sys
import re
import argparse
from typing import Iterator, List, Optional
import os

class LargeFileAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        
    def get_file_info(self, estimate_lines: bool = True) -> dict:
        """Get basic file information with optional line estimation"""
        line_count = "estimated"
        
        if estimate_lines and self.file_size > 100 * 1024 * 1024:  # > 100MB
            # Estimate lines by sampling first 10,000 lines
            sample_lines = 0
            sample_bytes = 0
            
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    sample_lines += 1
                    sample_bytes += len(line.encode('utf-8'))
                    if sample_lines >= 10000:
                        break
            
            if sample_bytes > 0:
                estimated_lines = int((self.file_size / sample_bytes) * sample_lines)
                line_count = f"~{estimated_lines:,} (estimated)"
        else:
            # Only count lines for smaller files
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for _ in f)
        
        return {
            'size_bytes': self.file_size,
            'size_mb': self.file_size / (1024 * 1024),
            'line_count': line_count,
            'filepath': self.filepath
        }
    
    def extract_lines_with_pattern(self, pattern: str, max_lines: int = 100) -> List[str]:
        """Extract lines matching a pattern"""
        matches = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if regex.search(line):
                    matches.append(f"Line {line_num}: {line.strip()}")
                    if len(matches) >= max_lines:
                        break
        
        return matches
    
    def extract_section(self, start_pattern: str, end_pattern: str, max_sections: int = 5) -> List[str]:
        """Extract sections between start and end patterns"""
        sections = []
        current_section = []
        in_section = False
        
        start_regex = re.compile(start_pattern, re.IGNORECASE)
        end_regex = re.compile(end_pattern, re.IGNORECASE)
        
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if start_regex.search(line):
                    in_section = True
                    current_section = [f"=== Section starting at line {line_num} ==="]
                    current_section.append(line.strip())
                elif end_regex.search(line) and in_section:
                    current_section.append(line.strip())
                    current_section.append("=== Section end ===\n")
                    sections.append('\n'.join(current_section))
                    current_section = []
                    in_section = False
                    
                    if len(sections) >= max_sections:
                        break
                elif in_section:
                    current_section.append(line.strip())
        
        return sections
    
    def get_summary_lines(self, line_interval: int = 100) -> List[str]:
        """Get every nth line as a summary"""
        summary_lines = []
        
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % line_interval == 0:
                    summary_lines.append(f"Line {line_num}: {line.strip()}")
        
        return summary_lines
    
    def safe_sample_file(self, sample_size: int = 50) -> List[str]:
        """Safely sample lines from very large files with strict size limits"""
        # For extremely large files (>500MB), be very conservative
        if self.file_size > 500 * 1024 * 1024:  # >500MB
            sample_size = min(sample_size, 30)  # Max 30 lines for huge files
        elif self.file_size > 100 * 1024 * 1024:  # >100MB
            sample_size = min(sample_size, 100)  # Max 100 lines for large files
        
        if self.file_size < 10 * 1024 * 1024:  # < 10MB, read normally
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return [line.strip()[:200] for line in f.readlines()[:sample_size]]  # Truncate long lines
        
        # For large files, sample from beginning, middle, and end
        sample_lines = []
        lines_per_section = max(1, sample_size // 3)
        
        # Sample from beginning
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i < lines_per_section:
                    # Truncate very long lines to prevent memory issues
                    truncated = line.strip()[:200]
                    sample_lines.append(f"[START-{i+1}] {truncated}")
                else:
                    break
        
        # Sample from middle (seek to middle of file)
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(self.file_size // 2)
            f.readline()  # Skip partial line
            for i, line in enumerate(f):
                if i < lines_per_section:
                    truncated = line.strip()[:200]
                    sample_lines.append(f"[MIDDLE-{i+1}] {truncated}")
                else:
                    break
        
        # Sample from end (approximate)
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(max(0, self.file_size - 1024 * 1024))  # Last 1MB
            f.readline()  # Skip partial line
            end_lines = []
            for line in f:
                end_lines.append(line.strip()[:200])  # Truncate long lines
            
            # Take last lines_per_section lines
            for i, line in enumerate(end_lines[-lines_per_section:]):
                sample_lines.append(f"[END-{i+1}] {line}")
        
        return sample_lines
    
    def extract_key_sections(self) -> dict:
        """Extract common key sections from NWTN output with conservative limits"""
        key_sections = {}
        
        # Adjust limits based on file size
        if self.file_size > 500 * 1024 * 1024:  # >500MB
            max_lines_per_section = 5
        elif self.file_size > 100 * 1024 * 1024:  # >100MB
            max_lines_per_section = 10
        else:
            max_lines_per_section = 20
        
        # Look for common patterns in NWTN output
        patterns = {
            'errors': r'error|exception|traceback|failed',
            'warnings': r'warning|warn',
            'results': r'result|output|conclusion',
            'performance': r'performance|time|speed|benchmark',
            'test': r'test|testing|validation',
            'reasoning': r'reasoning|logic|inference',
            'meta': r'meta.*reasoning|meta.*analysis'
        }
        
        for section_name, pattern in patterns.items():
            matches = self.extract_lines_with_pattern(pattern, max_lines=max_lines_per_section)
            if matches:
                key_sections[section_name] = matches
        
        return key_sections
    
    def ultra_safe_preview(self) -> dict:
        """Ultra-safe preview for extremely large files (>500MB)"""
        if self.file_size > 500 * 1024 * 1024:
            # For files >500MB, only return basic stats and tiny samples
            info = self.get_file_info()
            
            # Get just 3 lines from start, middle, end
            preview_lines = []
            
            # First line
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                if first_line:
                    preview_lines.append(f"[FIRST] {first_line.strip()[:100]}")
            
            # Middle line
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.file_size // 2)
                f.readline()  # Skip partial line
                middle_line = f.readline()
                if middle_line:
                    preview_lines.append(f"[MIDDLE] {middle_line.strip()[:100]}")
            
            # Last line
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(max(0, self.file_size - 1024))  # Last 1KB
                lines = f.readlines()
                if lines:
                    last_line = lines[-1]
                    preview_lines.append(f"[LAST] {last_line.strip()[:100]}")
            
            return {
                'file_info': info,
                'preview_lines': preview_lines,
                'warning': f'File is {info["size_mb"]:.1f}MB - showing minimal preview only'
            }
        else:
            # For smaller files, use normal safe_sample_file
            return {
                'file_info': self.get_file_info(),
                'sample_lines': self.safe_sample_file(20)
            }

def main():
    parser = argparse.ArgumentParser(description='Analyze large text files safely')
    parser.add_argument('filepath', help='Path to the large file')
    parser.add_argument('--info', action='store_true', help='Show file information')
    parser.add_argument('--pattern', help='Extract lines matching pattern')
    parser.add_argument('--start-pattern', help='Start pattern for section extraction')
    parser.add_argument('--end-pattern', help='End pattern for section extraction')
    parser.add_argument('--summary', type=int, default=100, help='Get every nth line as summary')
    parser.add_argument('--key-sections', action='store_true', help='Extract key sections')
    parser.add_argument('--max-lines', type=int, default=100, help='Maximum lines to extract')
    parser.add_argument('--safe-sample', type=int, default=50, help='Safely sample lines from large files')
    parser.add_argument('--ultra-safe', action='store_true', help='Ultra-safe preview for extremely large files (>500MB)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File {args.filepath} does not exist")
        sys.exit(1)
    
    analyzer = LargeFileAnalyzer(args.filepath)
    
    if args.info:
        info = analyzer.get_file_info()
        print(f"File: {info['filepath']}")
        print(f"Size: {info['size_mb']:.2f} MB ({info['size_bytes']:,} bytes)")
        print(f"Lines: {info['line_count']}")
    
    if args.pattern:
        matches = analyzer.extract_lines_with_pattern(args.pattern, args.max_lines)
        print(f"\nLines matching '{args.pattern}':")
        for match in matches:
            print(match)
    
    if args.start_pattern and args.end_pattern:
        sections = analyzer.extract_section(args.start_pattern, args.end_pattern)
        print(f"\nSections between '{args.start_pattern}' and '{args.end_pattern}':")
        for section in sections:
            print(section)
    
    if args.key_sections:
        key_sections = analyzer.extract_key_sections()
        print("\nKey sections found:")
        for section_name, matches in key_sections.items():
            print(f"\n=== {section_name.upper()} ===")
            for match in matches:
                print(match)
    
    if args.ultra_safe:
        preview = analyzer.ultra_safe_preview()
        print(f"\nUltra-safe preview:")
        print(f"File: {preview['file_info']['filepath']}")
        print(f"Size: {preview['file_info']['size_mb']:.2f} MB")
        print(f"Lines: {preview['file_info']['line_count']}")
        if 'warning' in preview:
            print(f"Warning: {preview['warning']}")
        print("\nPreview lines:")
        for line in preview.get('preview_lines', preview.get('sample_lines', [])):
            print(line)
    
    if args.safe_sample and not args.ultra_safe:
        sample = analyzer.safe_sample_file(args.safe_sample)
        print(f"\nSafe sample ({len(sample)} lines):")
        for line in sample:
            print(line)
    
    if not any([args.info, args.pattern, args.start_pattern, args.key_sections, args.safe_sample, args.ultra_safe]):
        summary = analyzer.get_summary_lines(args.summary)
        print(f"\nSummary (every {args.summary}th line):")
        for line in summary:
            print(line)

if __name__ == "__main__":
    main()