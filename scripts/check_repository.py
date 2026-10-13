#!/usr/bin/env python3
"""Offline syntax and regression checks; never imports hardware entry points."""
import ast
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('evera', 'robot', 'training', 'pi05', 'vision', 'vision_training',
               'vision_dashboard', 'scripts', 'tests')


def main():
    for executable in ('bash', 'node'):
        if not shutil.which(executable):
            raise SystemExit(f'Required check tool is missing: {executable}')
    counts = {'Python': 0, 'Shell': 0, 'JavaScript': 0}
    for directory in SOURCE_DIRS:
        for path in sorted((ROOT / directory).rglob('*')):
            if not path.is_file() or 'vendor' in path.parts or '__pycache__' in path.parts:
                continue
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=str(path))
                counts['Python'] += 1
            elif path.suffix == '.sh':
                subprocess.run(['bash', '-n', str(path)], check=True)
                counts['Shell'] += 1
            elif path.suffix in ('.js', '.mjs'):
                subprocess.run(['node', '--input-type=module', '--check'],
                               input=path.read_text(), text=True, check=True)
                counts['JavaScript'] += 1
    print(f'Syntax checks passed: {counts}', flush=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, check=True)
    subprocess.run(['node', '--test', 'tests/backend.test.mjs'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
