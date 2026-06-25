import argparse
import json
from pathlib import Path


def metric_keys(metrics):
    if not metrics:
        return ""
    keys = sorted(metrics.keys())
    return " [" + ", ".join(keys[:8]) + (" ..." if len(keys) > 8 else "") + "]"


def walk(node, depth, max_depth):
    if depth > max_depth:
        return
    frame = node.get("frame", {})
    name = frame.get("name", "<unknown>")
    metrics = node.get("metrics", {})
    print("  " * depth + f"- {name}{metric_keys(metrics)}")
    for child in node.get("children", []):
        walk(child, depth + 1, max_depth)


def main():
    parser = argparse.ArgumentParser(description="Print a compact tree from a Proton hatchet JSON file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    data = json.loads(args.path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "traceEvents" in data:
        print(f"chrome trace events: {len(data['traceEvents'])}")
        for event in data["traceEvents"][:20]:
            print(f"- {event.get('name')} cat={event.get('cat')} dur={event.get('dur')}")
        return

    if not isinstance(data, list):
        raise TypeError(f"Expected a Proton tree list or Chrome trace dict, got {type(data).__name__}")

    for i, root in enumerate(data):
        print(f"root[{i}]")
        if isinstance(root, dict) and "frame" in root:
            walk(root, 0, args.depth)
        elif isinstance(root, dict):
            print(json.dumps(root, indent=2)[:1200])
        else:
            print(repr(root)[:1200])


if __name__ == "__main__":
    main()
