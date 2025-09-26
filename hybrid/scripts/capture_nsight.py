#!/usr/bin/env python3
"""Convenience wrapper for Nsight Systems/Compute profiling.

Usage:
  python scripts/capture_nsight.py --mode systems --output traces/run1 \
      -- python -m hrm_lm.training.train --config ...

The script checks tool availability, builds the profiling command, and
executes it in a subprocess. Outputs are stored with the provided base
path (Nsight adds appropriate extensions).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_tool_exists(name: str) -> None:
  if shutil.which(name) is None:
    raise RuntimeError(f"Required tool '{name}' not found on PATH. Install Nsight before running this command.")


def run_command(cmd: list[str]) -> None:
  proc = subprocess.run(cmd, check=False)
  if proc.returncode != 0:
    raise RuntimeError(f"Command {' '.join(cmd)} exited with code {proc.returncode}")


def main() -> None:
  parser = argparse.ArgumentParser(description='Profile a command with Nsight Systems and/or Nsight Compute.')
  parser.add_argument('--mode', choices=['systems', 'compute', 'both'], default='systems', help='Select which profiler to launch.')
  parser.add_argument('--output', required=True, help='Output base path for profiler traces (extensions are added automatically).')
  parser.add_argument('--nsys-args', default='--trace=cuda,nvtx', help='Additional Nsight Systems arguments.')
  parser.add_argument('--ncu-args', default='--set full', help='Additional Nsight Compute arguments.')
  parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Command to profile (prefix with -- to separate).')
  args = parser.parse_args()

  if not args.cmd:
    parser.error('Supply the command to profile after --')

  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  if args.mode in {'systems', 'both'}:
    ensure_tool_exists('nsys')
    nsys_cmd = ['nsys', 'profile', '-o', str(output_path), *args.nsys_args.split(), '--'] + args.cmd
    run_command(nsys_cmd)

  if args.mode in {'compute', 'both'}:
    ensure_tool_exists('ncu')
    ncu_cmd = ['ncu', *args.ncu_args.split(), '--target-processes', 'all', '-o', str(output_path), '--'] + args.cmd
    run_command(ncu_cmd)


if __name__ == '__main__':
  try:
    main()
  except Exception as exc:  # pragma: no cover - CLI error handling
    sys.stderr.write(f'error: {exc}\n')
    sys.exit(1)
