import argparse, subprocess, pathlib, os

def parse_args() -> argparse.Namespace:
    options = 'run', 'debug',
    parser = argparse.ArgumentParser()
    parser.add_argument('args', choices=options)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.args == 'run':
        print('run')
        # path = pathlib.Path('.') / 'src' / 'main' / 'main' / 'main.py'
        # subprocess.run(f'python {path.absolute()}', shell=True)
    
    if args.args == 'debug':
        print('debug')

if __name__ == '__main__':
    main()
