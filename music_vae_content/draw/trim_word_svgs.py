#!/usr/bin/env python3
"""Trim nodes/word SVGs to outer component bounds and export 1:1 PNGs."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DRAW_DIR = Path(__file__).resolve().parent
WORD_DIR = DRAW_DIR.parent / "nodes" / "word"

# Minimal margin so card borders are not clipped by the viewBox / PNG export.
MM_PADDING = 1.0
PX_PER_MM = 96 / 25.4


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = value.strip()
    return float(value) if value else default


def parse_transform(transform: str | None) -> tuple[float, float]:
    if not transform:
        return 0.0, 0.0
    match = re.search(r"translate\(\s*([^,\)]+)(?:[\s,]+([^,\)]+))?\s*\)", transform)
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2) or 0.0)


def rect_bbox(rect: ET.Element, tx: float, ty: float) -> BBox:
    x = parse_float(rect.get("x")) + tx
    y = parse_float(rect.get("y")) + ty
    w = parse_float(rect.get("width"))
    h = parse_float(rect.get("height"))
    stroke = parse_float(rect.get("stroke-width"), 0.0)
    pad = stroke / 2
    return BBox(x - pad, y - pad, x + w + pad, y + h + pad)


def collect_stroked_rects(
    element: ET.Element,
    rects: list[tuple[BBox, float]],
    tx: float,
    ty: float,
) -> None:
    tag = element.tag.split("}")[-1]
    if tag == "g":
        gtx, gty = parse_transform(element.get("transform"))
        for child in element:
            collect_stroked_rects(child, rects, tx + gtx, ty + gty)
        return
    if tag == "rect" and element.get("stroke"):
        rects.append((rect_bbox(element, tx, ty), parse_float(element.get("stroke-width"), 0.0)))


def collect_union_bounds(
    element: ET.Element,
    bbox: BBox | None,
    tx: float,
    ty: float,
) -> BBox:
    tag = element.tag.split("}")[-1]

    if tag == "g":
        gtx, gty = parse_transform(element.get("transform"))
        for child in element:
            bbox = collect_union_bounds(child, bbox, tx + gtx, ty + gty)
        return bbox

    if tag == "rect" and element.get("stroke"):
        rb = rect_bbox(element, tx, ty)
        if bbox is None:
            return rb
        return BBox(
            min(bbox.xmin, rb.xmin),
            min(bbox.ymin, rb.ymin),
            max(bbox.xmax, rb.xmax),
            max(bbox.ymax, rb.ymax),
        )

    return bbox


def compute_bounds(root: ET.Element) -> BBox:
    stroked: list[BBox] = []
    for child in root:
        if child.tag.split("}")[-1] in {"defs", "style"}:
            continue
        collect_stroked_rects(child, stroked, 0.0, 0.0)

    if stroked:
        boxes = [bbox for bbox, _ in stroked]
        outer = max(boxes, key=lambda b: b.area)
        union = boxes[0]
        for box in boxes[1:]:
            union = BBox(
                min(union.xmin, box.xmin),
                min(union.ymin, box.ymin),
                max(union.xmax, box.xmax),
                max(union.ymax, box.ymax),
            )
        # Multi-panel nodes (e.g. stacked bars): use full union.
        if len(boxes) >= 2 and outer.area < 0.8 * union.area:
            return union
        return outer

    union: BBox | None = None
    for child in root:
        if child.tag.split("}")[-1] in {"defs", "style"}:
            continue
        union = collect_union_bounds(child, union, 0.0, 0.0)
    if union is None:
        raise RuntimeError("No bounds found")
    return union


def pad_bbox(bbox: BBox, mm: float = MM_PADDING) -> BBox:
    pad = mm * PX_PER_MM
    return BBox(bbox.xmin - pad, bbox.ymin - pad, bbox.xmax + pad, bbox.ymax + pad)


def remove_canvas_background(root: ET.Element) -> None:
    style = root.get("style", "")
    if style:
        style = re.sub(r"background-color:\s*[^;]+;?\s*", "", style).strip()
        if style:
            root.set("style", style)
        else:
            root.attrib.pop("style", None)

    for child in list(root):
        if child.tag.split("}")[-1] != "rect" or child.get("stroke"):
            continue
        fill = (child.get("fill") or "").lower()
        if fill not in {"#ffffff", "#fff", "white"}:
            continue
        if parse_float(child.get("x")) == 0.0 and parse_float(child.get("y")) == 0.0:
            if parse_float(child.get("width")) > 100 and parse_float(child.get("height")) > 100:
                root.remove(child)


def trim_svg(svg_path: Path, out_svg: Path) -> tuple[int, int]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    bbox = pad_bbox(compute_bounds(root))
    remove_canvas_background(root)

    width = round(bbox.width)
    height = round(bbox.height)
    root.set("viewBox", f"{bbox.xmin:g} {bbox.ymin:g} {bbox.width:g} {bbox.height:g}")
    root.set("width", str(width))
    root.set("height", str(height))
    root.attrib.pop("preserveAspectRatio", None)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree.write(out_svg, encoding="utf-8", xml_declaration=True)
    return width, height


def export_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    subprocess.run(
        [
            "rsvg-convert",
            str(svg_path),
            "-o",
            str(png_path),
            "-w",
            str(width),
            "-h",
            str(height),
            "-b",
            "none",
        ],
        check=True,
    )


def main() -> None:
    svgs = sorted(WORD_DIR.glob("*.svg"))
    print(f"Trimming {len(svgs)} SVGs from {WORD_DIR.name}/ -> {DRAW_DIR.name}/")
    for src in svgs:
        out_svg = DRAW_DIR / src.name
        out_png = DRAW_DIR / f"{src.stem}.png"
        w, h = trim_svg(src, out_svg)
        export_png(out_svg, out_png, w, h)
        print(f"  {src.name}: {w}x{h}")
    print("Done.")


if __name__ == "__main__":
    main()
