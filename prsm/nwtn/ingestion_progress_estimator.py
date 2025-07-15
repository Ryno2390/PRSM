#!/usr/bin/env python3
"""
Ingestion Progress Estimator
===========================

This tool estimates completion time for the ingestion process by analyzing
file creation patterns and processing rates.
"""

import os
import time
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import glob
import re

class IngestionProgressEstimator:
    """Estimate ingestion progress and completion time"""
    
    def __init__(self):
        self.storage_path = Path("/Volumes/My Passport/PRSM_Storage")
        self.content_path = self.storage_path / "PRSM_Content"
        self.metadata_path = self.storage_path / "PRSM_Metadata"
        self.log_path = Path("/tmp/ingestion.log")
        self.target_items = 150000  # Target number of items
        
    def analyze_file_progress(self) -> Dict[str, Any]:
        """Analyze progress based on stored files"""
        
        results = {
            "file_analysis": {},
            "processing_rate": {},
            "time_estimates": {},
            "current_status": {}
        }
        
        # Count files by creation time
        content_files = self._get_files_with_timestamps(self.content_path)
        metadata_files = self._get_files_with_timestamps(self.metadata_path)
        
        results["file_analysis"] = {
            "content_files": len(content_files),
            "metadata_files": len(metadata_files),
            "total_files": len(content_files) + len(metadata_files),
            "oldest_file": min([f["created"] for f in content_files]) if content_files else None,
            "newest_file": max([f["created"] for f in content_files]) if content_files else None
        }
        
        # Calculate processing rates
        if content_files:
            results["processing_rate"] = self._calculate_processing_rate(content_files)
            results["time_estimates"] = self._estimate_completion_time(content_files)
        
        return results
    
    def analyze_log_progress(self) -> Dict[str, Any]:
        """Analyze progress from log file"""
        
        if not self.log_path.exists():
            return {"log_analysis": "Log file not found"}
        
        try:
            # Read recent log entries
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
            
            # Find key patterns
            start_time = self._find_start_time(lines)
            processing_counts = self._count_processing_activity(lines)
            quality_stats = self._analyze_quality_decisions(lines)
            error_patterns = self._analyze_errors(lines)
            
            return {
                "log_analysis": {
                    "start_time": start_time,
                    "processing_counts": processing_counts,
                    "quality_stats": quality_stats,
                    "error_patterns": error_patterns,
                    "total_log_lines": len(lines)
                }
            }
        except Exception as e:
            return {"log_analysis": f"Error analyzing log: {e}"}
    
    def get_comprehensive_estimate(self) -> Dict[str, Any]:
        """Get comprehensive progress estimate"""
        
        file_progress = self.analyze_file_progress()
        log_progress = self.analyze_log_progress()
        
        # Combine analyses
        comprehensive = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_based_analysis": file_progress,
            "log_based_analysis": log_progress,
            "summary": self._generate_summary(file_progress, log_progress)
        }
        
        return comprehensive
    
    def _get_files_with_timestamps(self, directory: Path) -> List[Dict[str, Any]]:
        """Get files with creation timestamps"""
        
        files = []
        if not directory.exists():
            return files
        
        for file_path in directory.rglob("*.dat"):
            try:
                stat = file_path.stat()
                files.append({
                    "path": str(file_path),
                    "created": stat.st_ctime,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            except (OSError, IOError):
                continue
        
        return sorted(files, key=lambda x: x["created"])
    
    def _calculate_processing_rate(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate processing rate from file creation times"""
        
        if len(files) < 2:
            return {"rate_available": False}
        
        # Calculate time spans
        start_time = files[0]["created"]
        end_time = files[-1]["created"]
        total_duration = end_time - start_time
        
        if total_duration <= 0:
            return {"rate_available": False}
        
        total_files = len(files)
        
        # Calculate rates
        files_per_second = total_files / total_duration
        files_per_minute = files_per_second * 60
        files_per_hour = files_per_second * 3600
        
        # Calculate recent rate (last 100 files)
        recent_files = files[-100:] if len(files) >= 100 else files
        if len(recent_files) >= 2:
            recent_duration = recent_files[-1]["created"] - recent_files[0]["created"]
            recent_rate_per_hour = (len(recent_files) / recent_duration) * 3600 if recent_duration > 0 else 0
        else:
            recent_rate_per_hour = files_per_hour
        
        return {
            "rate_available": True,
            "total_duration_hours": total_duration / 3600,
            "files_per_second": files_per_second,
            "files_per_minute": files_per_minute,
            "files_per_hour": files_per_hour,
            "recent_rate_per_hour": recent_rate_per_hour,
            "total_files_processed": total_files
        }
    
    def _estimate_completion_time(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate completion time"""
        
        if len(files) < 2:
            return {"estimation_available": False}
        
        # Use recent rate for estimation
        processing_rate = self._calculate_processing_rate(files)
        
        if not processing_rate["rate_available"]:
            return {"estimation_available": False}
        
        current_count = len(files)
        remaining_items = self.target_items - current_count
        
        # Use recent rate for more accurate prediction
        recent_rate = processing_rate["recent_rate_per_hour"]
        
        if recent_rate <= 0:
            return {"estimation_available": False}
        
        # Calculate time estimates
        hours_remaining = remaining_items / recent_rate
        completion_time = datetime.now(timezone.utc) + timedelta(hours=hours_remaining)
        
        # Calculate progress percentage
        progress_percent = (current_count / self.target_items) * 100
        
        return {
            "estimation_available": True,
            "current_items": current_count,
            "target_items": self.target_items,
            "remaining_items": remaining_items,
            "progress_percent": progress_percent,
            "recent_rate_per_hour": recent_rate,
            "estimated_hours_remaining": hours_remaining,
            "estimated_completion_time": completion_time.isoformat(),
            "estimated_completion_local": completion_time.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        }
    
    def _find_start_time(self, lines: List[str]) -> str:
        """Find ingestion start time from log"""
        
        for line in lines:
            if "Starting large-scale breadth-optimized ingestion" in line:
                # Extract timestamp
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if match:
                    return match.group(1)
        
        return "Start time not found"
    
    def _count_processing_activity(self, lines: List[str]) -> Dict[str, int]:
        """Count processing activities from log"""
        
        counts = {
            "papers_downloaded": 0,
            "quality_assessments": 0,
            "batch_processing": 0,
            "storage_operations": 0,
            "errors": 0
        }
        
        for line in lines:
            if "Quality assessment completed" in line:
                counts["quality_assessments"] += 1
            elif "Batch Processing" in line:
                counts["batch_processing"] += 1
            elif "Storing content" in line:
                counts["storage_operations"] += 1
            elif "error" in line.lower():
                counts["errors"] += 1
            elif any(pattern in line for pattern in ["2507.", "2506.", "arxiv"]):
                counts["papers_downloaded"] += 1
        
        return counts
    
    def _analyze_quality_decisions(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze quality filter decisions"""
        
        decisions = {"accept": 0, "reject": 0, "review": 0}
        quality_scores = []
        
        for line in lines:
            if "decision=" in line:
                if "decision=accept" in line:
                    decisions["accept"] += 1
                elif "decision=reject" in line:
                    decisions["reject"] += 1
                elif "decision=review" in line:
                    decisions["review"] += 1
                
                # Extract quality score
                score_match = re.search(r'overall_quality=([0-9.]+)', line)
                if score_match:
                    quality_scores.append(float(score_match.group(1)))
        
        total_decisions = sum(decisions.values())
        
        return {
            "decisions": decisions,
            "total_decisions": total_decisions,
            "acceptance_rate": decisions["accept"] / total_decisions if total_decisions > 0 else 0,
            "rejection_rate": decisions["reject"] / total_decisions if total_decisions > 0 else 0,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "quality_score_range": {
                "min": min(quality_scores) if quality_scores else 0,
                "max": max(quality_scores) if quality_scores else 0
            }
        }
    
    def _analyze_errors(self, lines: List[str]) -> Dict[str, int]:
        """Analyze error patterns"""
        
        error_patterns = {
            "storage_errors": 0,
            "quality_errors": 0,
            "network_errors": 0,
            "general_errors": 0
        }
        
        for line in lines:
            if "error" in line.lower():
                if "storage" in line.lower() or "database" in line.lower():
                    error_patterns["storage_errors"] += 1
                elif "quality" in line.lower() or "assessment" in line.lower():
                    error_patterns["quality_errors"] += 1
                elif "network" in line.lower() or "connection" in line.lower():
                    error_patterns["network_errors"] += 1
                else:
                    error_patterns["general_errors"] += 1
        
        return error_patterns
    
    def _generate_summary(self, file_progress: Dict[str, Any], log_progress: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of progress"""
        
        summary = {
            "ingestion_active": True,
            "status": "running",
            "primary_metric": "file_count"
        }
        
        # File-based summary
        if file_progress.get("file_analysis", {}).get("content_files", 0) > 0:
            file_count = file_progress["file_analysis"]["content_files"]
            summary["current_items"] = file_count
            
            if file_progress.get("time_estimates", {}).get("estimation_available"):
                est = file_progress["time_estimates"]
                summary.update({
                    "progress_percent": est["progress_percent"],
                    "estimated_hours_remaining": est["estimated_hours_remaining"],
                    "estimated_completion": est["estimated_completion_local"],
                    "current_rate_per_hour": est["recent_rate_per_hour"]
                })
        
        # Log-based enhancements
        if log_progress.get("log_analysis", {}).get("quality_stats"):
            quality = log_progress["log_analysis"]["quality_stats"]
            summary["quality_stats"] = {
                "acceptance_rate": quality["acceptance_rate"],
                "total_assessed": quality["total_decisions"],
                "average_quality": quality["average_quality_score"]
            }
        
        return summary
    
    def display_progress_report(self):
        """Display comprehensive progress report"""
        
        analysis = self.get_comprehensive_estimate()
        
        print("🚀 PRSM NWTN - INGESTION PROGRESS ESTIMATION")
        print("=" * 70)
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # File Progress
        file_analysis = analysis["file_based_analysis"]["file_analysis"]
        print("📁 FILE PROGRESS:")
        print(f"   📄 Content Files: {file_analysis['content_files']:,}")
        print(f"   🏷️  Metadata Files: {file_analysis['metadata_files']:,}")
        print(f"   📊 Total Files: {file_analysis['total_files']:,}")
        print()
        
        # Processing Rate
        if analysis["file_based_analysis"].get("processing_rate", {}).get("rate_available"):
            rate = analysis["file_based_analysis"]["processing_rate"]
            print("⚡ PROCESSING RATE:")
            print(f"   🔄 Current Rate: {rate['recent_rate_per_hour']:.1f} items/hour")
            print(f"   📈 Overall Rate: {rate['files_per_hour']:.1f} items/hour")
            print(f"   ⏱️  Running Time: {rate['total_duration_hours']:.1f} hours")
            print()
        
        # Time Estimates
        if analysis["file_based_analysis"].get("time_estimates", {}).get("estimation_available"):
            est = analysis["file_based_analysis"]["time_estimates"]
            print("🎯 COMPLETION ESTIMATES:")
            print(f"   📊 Progress: {est['progress_percent']:.2f}% ({est['current_items']:,} / {est['target_items']:,})")
            print(f"   ⏳ Estimated Remaining: {est['estimated_hours_remaining']:.1f} hours")
            print(f"   🏁 Estimated Completion: {est['estimated_completion_local']}")
            print(f"   📉 Remaining Items: {est['remaining_items']:,}")
            print()
        
        # Quality Stats
        if analysis["log_based_analysis"].get("log_analysis", {}).get("quality_stats"):
            quality = analysis["log_based_analysis"]["log_analysis"]["quality_stats"]
            print("🔍 QUALITY FILTER STATS:")
            print(f"   ✅ Acceptance Rate: {quality['acceptance_rate']:.1%}")
            print(f"   📊 Total Assessed: {quality['total_decisions']:,}")
            print(f"   🎯 Average Quality: {quality['average_quality_score']:.3f}")
            print(f"   📈 Quality Range: {quality['quality_score_range']['min']:.3f} - {quality['quality_score_range']['max']:.3f}")
            print()
        
        # System Status
        print("🏥 SYSTEM STATUS:")
        print(f"   🔄 Status: Active and Processing")
        print(f"   📝 Log Entries: {analysis['log_based_analysis'].get('log_analysis', {}).get('total_log_lines', 0):,}")
        
        # Error Summary
        if analysis["log_based_analysis"].get("log_analysis", {}).get("error_patterns"):
            errors = analysis["log_based_analysis"]["log_analysis"]["error_patterns"]
            total_errors = sum(errors.values())
            if total_errors > 0:
                print(f"   ⚠️  Total Errors: {total_errors:,} (system continues running)")
        
        print()
        print("=" * 70)
        print("📝 Note: Estimates based on recent processing rate")
        print("🔄 Run this script again later for updated estimates")
        print("=" * 70)


def main():
    """Main function"""
    estimator = IngestionProgressEstimator()
    estimator.display_progress_report()


if __name__ == "__main__":
    main()