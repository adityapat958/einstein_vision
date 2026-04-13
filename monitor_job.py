#!/usr/bin/env python3
"""
Job monitoring utility for SBATCH submissions.
Usage: python3 monitor_job.py <job_id>
       python3 monitor_job.py <job_id> --follow  # continuous monitoring
       python3 monitor_job.py <job_id> --output  # show current output
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path

def get_job_status(job_id):
    """Get status of SLURM job."""
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].split()[4]  # Job status column
        return "COMPLETED/NOT_FOUND"
    except Exception as e:
        return f"ERROR: {e}"

def get_log_files(job_id):
    """Find job log files (*.out and *.err)."""
    pattern = f"rcnn_lane_detection_{job_id}"
    cwd = Path.home() / "repos" / "cv" / "p3"
    
    out_file = None
    err_file = None
    
    for f in cwd.glob(f"{pattern}.out"):
        out_file = f
    for f in cwd.glob(f"{pattern}.err"):
        err_file = f
    
    return out_file, err_file

def read_file_tail(filepath, num_lines=20):
    """Read last N lines of a file."""
    try:
        if not filepath or not filepath.exists():
            return None
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return ''.join(lines[-num_lines:])
    except Exception as e:
        return f"ERROR reading file: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Monitor RCNN lane detection job status"
    )
    parser.add_argument(
        'job_id',
        type=int,
        help='SLURM job ID'
    )
    parser.add_argument(
        '--follow',
        action='store_true',
        help='Continuously monitor job (updates every 10 seconds)'
    )
    parser.add_argument(
        '--output',
        action='store_true',
        help='Show last 30 lines of output file'
    )
    parser.add_argument(
        '--error',
        action='store_true',
        help='Show last 30 lines of error file'
    )
    parser.add_argument(
        '--tail',
        type=int,
        default=20,
        help='Number of lines to show (default: 20)'
    )
    parser.add_argument(
        '--results',
        action='store_true',
        help='Show processing results summary'
    )
    
    args = parser.parse_args()
    job_id = args.job_id
    
    print(f"Monitoring Job ID: {job_id}")
    print(f"Job Details: https://hpc.wpi.edu/slurm/")
    print("-" * 60)
    
    out_file, err_file = get_log_files(job_id)
    
    if not out_file:
        print(f"ERROR: Could not find log files for job {job_id}")
        print(f"Looking for files matching: rcnn_lane_detection_{job_id}.out/err")
        return 1
    
    print(f"Output file: {out_file}")
    print(f"Error file:  {err_file}")
    print("-" * 60)
    
    if args.follow:
        try:
            iteration = 0
            while True:
                os.system('clear')
                iteration += 1
                status = get_job_status(job_id)
                print(f"[{time.strftime('%H:%M:%S')}] Iteration {iteration} - Job Status: {status}")
                print("-" * 60)
                
                output = read_file_tail(out_file, args.tail)
                if output:
                    print("LATEST OUTPUT:")
                    print(output)
                    print("-" * 60)
                
                if status in ["COMPLETED", "FAILED", "CANCELLED", "NOT_FOUND"]:
                    print(f"\nJob has finished with status: {status}")
                    break
                
                print(f"\nMonitoring... (updating every 10 seconds, press Ctrl+C to stop)")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
    
    if args.output:
        print("LAST {} LINES OF OUTPUT:".format(args.tail))
        print("-" * 60)
        output = read_file_tail(out_file, args.tail)
        if output:
            print(output)
    
    if args.error:
        print("LAST {} LINES OF ERROR:".format(args.tail))
        print("-" * 60)
        if err_file:
            error = read_file_tail(err_file, args.tail)
            if error:
                print(error)
        else:
            print("No error file found.")
    
    if args.results:
        print("PROCESSING RESULTS:")
        print("-" * 60)
        results_dir = Path.home() / "repos" / "cv" / "p3" / "cv_p3" / "lane_detection_results"
        if results_dir.exists():
            log_file = results_dir / "processing_log.txt"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    print(f.read())
            
            mp4_files = list(results_dir.glob("lane_detection_scene*.mp4"))
            print(f"\nGenerated videos: {len(mp4_files)}/13")
            for f in sorted(mp4_files):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  ✓ {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"Results directory not found: {results_dir}")
    
    if not any([args.follow, args.output, args.error, args.results]):
        # Default: show status and last few lines
        status = get_job_status(job_id)
        print(f"Job Status: {status}")
        output = read_file_tail(out_file, 15)
        if output:
            print("\nLast 15 lines of output:")
            print(output)

if __name__ == "__main__":
    sys.exit(main())
