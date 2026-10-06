"""Headless entry point for the extracted MemoryMade generation core."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="MemoryMade recognition and 3D core; no web UI")
    parser.add_argument('--resource-root', type=Path, required=True,
                        help='Existing model/tool/environment resource directory')
    parser.add_argument('--output-root', type=Path, default=Path(__file__).parent / 'runs')
    parser.add_argument('--asset-root', type=Path, help='Existing CPU vision assets directory')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    recognize = sub.add_parser('recognize')
    recognize.add_argument('image', type=Path)
    reconstruct = sub.add_parser('reconstruct')
    reconstruct.add_argument('image', type=Path)
    reconstruct.add_argument('--backend', choices=['triposr','hunyuan','hunyuan21','hunyuan21_pbr'], default='hunyuan21')
    reconstruct.add_argument('--width-mm', type=float, default=100)
    reconstruct.add_argument('--material', default='光敏树脂')
    reconstruct.add_argument('--color', default='#E9E4D8')
    reconstruct.add_argument('--quality', default='极精细（512 · 80步）')
    reconstruct.add_argument('--subject', choices=['object','human','building'], default='object')
    args = parser.parse_args()
    root = args.resource_root.resolve()
    if not root.is_dir():
        parser.error('Resource directory does not exist')
    if args.command != 'status' and not args.image.is_file():
        parser.error('Image does not exist')
    if args.command == 'reconstruct':
        import re
        if not 1 <= args.width_mm <= 1000 or not re.fullmatch(r'#[0-9a-fA-F]{6}', args.color):
            parser.error('Width must be 1–1000 mm and color must be #RRGGBB')
    os.environ['MEMORYMADE_RESOURCE_ROOT'] = str(root)
    os.environ['MEMORYMADE_WRITE_ROOT'] = str(args.output_root.resolve())
    os.environ['MEMORYMADE_DATA_DIR'] = str(args.output_root.resolve() / 'data')
    if args.asset_root:
        os.environ['MEMORYMADE_ASSET_DIR'] = str(args.asset_root.resolve())
    from memorymade.ai_stack import analyze_object, generate_local_3d, stack_status
    if args.command == 'status':
        result = stack_status()
    elif args.command == 'recognize':
        result = analyze_object(str(args.image.resolve()))
    else:
        from memorymade.config import MATERIALS
        from memorymade.fidelity import QUALITY_PROFILES
        if args.material not in MATERIALS or args.quality not in QUALITY_PROFILES:
            parser.error('Unknown material or quality profile')
        result = generate_local_3d([str(args.image.resolve())], 'MemoryMade', '', '',
                                  args.width_mm, args.material, args.color,
                                  quality=args.quality, backend=args.backend,
                                  subject_kind=args.subject).as_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
