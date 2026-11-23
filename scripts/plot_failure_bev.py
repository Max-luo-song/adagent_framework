#!/usr/bin/env python3
"""Plot full EGO and target bodies for one archived closed-loop failure.

Example:
    python scripts/plot_failure_bev.py \
        --archive /path/to/navtest_failures_R_complete \
        --token 4abb45d9ee1a5c5c \
        --out /tmp/bev.png

The output uses the official map and recorded EGO path. Every vehicle whose body
comes within the configurable radius of EGO during the selected frames is drawn
with its trajectory, complete oriented body and heading. The viewport is sized
from the complete bodies, so vehicle footprints are never cut off.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
from pyproj import Transformer
from shapely.geometry import Point, Polygon as ShapelyPolygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import structure_facts as sf  # noqa: E402


def frame_range(value, violation, frame_count, outcome=None):
    """Parse START:END; collisions stop at violation to avoid post-impact overlap."""
    if value:
        try:
            start_text, end_text = value.split(":", 1)
            start, end = int(start_text), int(end_text)
        except (AttributeError, TypeError, ValueError):
            raise argparse.ArgumentTypeError("frame range must be START:END") from None
    else:
        extra = 0 if outcome in ("collision", "collision_and_departure") else 1
        start, end = 0, min(violation + extra, frame_count - 1)
    if not 0 <= start <= end < frame_count:
        raise argparse.ArgumentTypeError(
            f"frame range {start}:{end} is outside recorded frames 0:{frame_count - 1}")
    return start, end


def projected_converter(frames, intersections):
    """Choose a metric CRS and convert UTM ego/actor coordinates into it."""
    extent = intersections.to_crs("EPSG:4326").total_bounds
    pose = np.asarray(frames[0]["ego2global"], dtype=float)
    lonlat = sf.utm_to_lonlat(float(pose[0, 3]), float(pose[1, 3]), extent)
    metric = sf.metric_crs(intersections.crs, *lonlat)
    transformer = Transformer.from_crs("EPSG:4326", metric, always_xy=True)

    def transform(x, y):
        return transformer.transform(*sf.utm_to_lonlat(float(x), float(y), extent))

    def heading(x, y, yaw):
        x0, y0 = transform(x, y)
        x1, y1 = transform(x + math.cos(yaw), y + math.sin(yaw))
        return x0, y0, math.atan2(y1 - y0, x1 - x0)

    return metric, transform, heading


def transformed_body(polygon, transform):
    points = polygon.exterior.coords if hasattr(polygon, "exterior") else polygon.points
    return np.asarray([transform(x, y) for x, y in points], dtype=float)


def body_center(points):
    return points[:-1].mean(axis=0)


def source_body_center(body):
    points = body.points if hasattr(body, "points") else body.exterior.coords
    return np.asarray(points, dtype=float)[:-1].mean(axis=0)


def draw_body(axis, points, heading, color, label, *, alpha=0.55, zorder=8,
              show_label=True, edgecolor="black", linewidth=0.75):
    axis.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=edgecolor,
                           linewidth=linewidth, alpha=alpha, zorder=zorder))
    center = body_center(points)
    length = 4.2
    axis.add_patch(FancyArrowPatch(
        tuple(center), (center[0] + length * math.cos(heading),
                        center[1] + length * math.sin(heading)),
        arrowstyle="-|>", mutation_scale=9, linewidth=0.75, color="black",
        zorder=zorder + 1))
    if show_label:
        axis.text(center[0], center[1] + 1.1, label, fontsize=6.5, color=edgecolor if edgecolor != "black" else color,
                  fontweight="bold" if edgecolor != "black" else "normal",
                  ha="center", va="bottom", zorder=zorder + 2)


def _actor_meta(frames, track, index):
    anns = frames[index].get("anns") or {}
    tokens = [str(t) for t in anns.get("track_tokens", [])]
    names = anns.get("gt_names", [])
    boxes = anns.get("gt_boxes", [])
    try:
        pos = tokens.index(track)
        box = boxes[pos]
        size_m = [round(float(box[3]), 2), round(float(box[4]), 2), round(float(box[5]), 2)]
        return str(names[pos]), size_m
    except (ValueError, IndexError, TypeError):
        return "vehicle", None


def _kinematics(frame_indices, times, points, yaws, violation):
    keep = [i for i, f in enumerate(frame_indices) if f <= violation] or list(range(len(frame_indices)))
    pts = [points[i] for i in keep]
    ts = [times[i] for i in keep]
    ys = [yaws[i] for i in keep]
    speeds = []
    disp = 0.0
    for i in range(1, len(pts)):
        d = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        dt = ts[i] - ts[i - 1]
        disp += d
        if dt > 1e-6:
            speeds.append(d / dt)
    unwrapped = [ys[0]]
    for y in ys[1:]:
        delta = (y - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi
        unwrapped.append(unwrapped[-1] + delta)
    yaw_deg = math.degrees(unwrapped[-1] - unwrapped[0])
    max_v = max(speeds) if speeds else 0.0
    return {
        "speed_start_mps": round(speeds[0], 2) if speeds else 0.0,
        "speed_violation_mps": round(speeds[-1], 2) if speeds else 0.0,
        "displacement_m": round(disp, 2),
        "yaw_change_deg": round(yaw_deg, 1),
        "is_stationary": bool(disp < 0.5 and max_v < 0.3),
    }


def map_layers(map_location):
    path = sf.map_file(map_location)
    return {
        name: sf.map_layer(gpd, path, name)
        for name in ("lanes_polygons", "baseline_paths", "road_segments", "intersections")
    }


def plot_failure_bev(archive, token, output, *, requested_range=None,
                     nearby_radius_m=30.0, highlight_track=None,
                     margin_m=8.0, dpi=150, title=None, violation_zoom_out=None,
                     violation_zoom_margin_m=7.0):
    archive = Path(archive).expanduser().resolve()
    frames, full_token = sf.scene_pickle(str(archive), token)
    outcome, official_token, violation = sf.outcome(str(archive), full_token)
    tracks = sf.actor_sequences(frames)
    start, end = frame_range(requested_range, violation, len(frames), outcome=outcome)
    indices = list(range(start, end + 1))

    layers = map_layers(frames[0]["map_location"])
    metric, transform, heading = projected_converter(frames, layers["intersections"])
    layers = {name: frame.to_crs(metric) for name, frame in layers.items()}
    ego_points = []
    body_geometry = []

    figure, axis = plt.subplots(figsize=(13, 9.5))
    layers["lanes_polygons"].plot(ax=axis, facecolor="#edf1f5", edgecolor="#9ca8b6",
                                  linewidth=0.55, zorder=0)
    layers["road_segments"].plot(ax=axis, facecolor="none", edgecolor="#596573",
                                  linewidth=0.8, zorder=1)
    layers["baseline_paths"].plot(ax=axis, color="#d5a21c", linewidth=0.55,
                                   alpha=0.75, zorder=2)
    layers["intersections"].boundary.plot(ax=axis, color="black", linewidth=1.6, zorder=3)

    t0 = frames[0]["timestamp"]
    frame_times = [(f["timestamp"] - t0) / 1e6 for f in frames]

    nearby = []
    for track, observations in sorted(tracks.items()):
        rows = {item["index"]: item for item in observations if start <= item["index"] <= end}
        if not rows:
            continue
        min_dist = math.inf
        viol_dist = None
        for index in rows:
            ego = sf.ego_body(frames[index], index)
            actor = sf.actor_body(rows[index], frames[index], index)
            distance = ego.distance(actor)
            if math.isfinite(distance):
                min_dist = min(min_dist, distance)
                if index == violation:
                    viol_dist = distance
        if min_dist <= nearby_radius_m:
            nearby.append((track, rows, min_dist, viol_dist))

    ego_color = "#f57c00"
    other_color = "#1976d2"
    ego_yaws = []
    ego_traj = []
    for index in indices:
        ego_utm = sf.ego_body(frames[index], index)
        ego_shape = transformed_body(ego_utm, transform)
        center = source_body_center(ego_utm)
        x, y, yaw = heading(center[0], center[1], sf.ego_heading(frames[index], index))
        is_viol = index == violation
        draw_body(axis, ego_shape, yaw, ego_color, f"E{index}",
                  alpha=0.78 if is_viol else 0.58, zorder=10 if is_viol else 8,
                  edgecolor="#ff1744" if is_viol else "black",
                  linewidth=2.2 if is_viol else 0.75)
        ego_points.append((x, y))
        ego_yaws.append(yaw)
        ego_traj.append({"frame": index, "t_s": round(frame_times[index], 2),
                         "x": round(float(x), 3), "y": round(float(y), 3), "yaw_rad": round(float(yaw), 4)})
        body_geometry.extend(ego_shape)

    ego_kinematics = _kinematics(indices, [frame_times[i] for i in indices], ego_points, ego_yaws, violation)
    ego_viol_yaw = ego_yaws[indices.index(violation)] if violation in indices else ego_yaws[-1]

    ego_points = np.asarray(ego_points)
    axis.plot(ego_points[:, 0], ego_points[:, 1], color=ego_color, linewidth=2.5, zorder=6)
    handles = [
        Line2D([0], [0], color=ego_color, linewidth=3.0, label="EGO"),
        Line2D([0], [0], color="#ff1744", linewidth=2.5, label=f"EGO violation (F{violation})"),
        Line2D([0], [0], color=other_color, linewidth=2.2, label=f"Other vehicles ({len(nearby)})"),
    ]
    actor_records = []
    actor_trajectories = {}
    for number, (track, rows, min_dist, viol_dist) in enumerate(nearby, start=1):
        obs_frames = sorted(rows)
        points, yaws, traj = [], [], []
        for index in obs_frames:
            item = rows[index]
            actor_utm = sf.actor_body(item, frames[index], index)
            actor_shape = transformed_body(actor_utm, transform)
            center = source_body_center(actor_utm)
            x, y, yaw = heading(center[0], center[1], item["world_yaw"])
            draw_body(axis, actor_shape, yaw, other_color, "", alpha=0.35, zorder=7, show_label=False)
            points.append((x, y))
            yaws.append(yaw)
            traj.append({"frame": index, "t_s": round(frame_times[index], 2),
                         "x": round(float(x), 3), "y": round(float(y), 3), "yaw_rad": round(float(yaw), 4)})
            body_geometry.extend(actor_shape)
        kin = _kinematics(obs_frames, [frame_times[i] for i in obs_frames], points, yaws, violation)
        rel_yaw = None
        if violation in obs_frames:
            v_yaw = yaws[obs_frames.index(violation)]
            rel_yaw = round(math.degrees((v_yaw - ego_viol_yaw + math.pi) % (2 * math.pi) - math.pi), 1)
        label_name, size_m = _actor_meta(frames, track, obs_frames[0])
        points = np.asarray(points)
        axis.plot(points[:, 0], points[:, 1], color=other_color,
                  linewidth=2.6 if track == highlight_track else 1.6,
                  alpha=0.9 if track == highlight_track else 0.7, zorder=6)
        actor_records.append({
            "vehicle": f"V{number}",
            "track": track,
            "label": label_name,
            "size_m": size_m,
            "frames_observed": obs_frames,
            "min_distance_m": round(float(min_dist), 2),
            "violation_distance_m": None if viol_dist is None else round(float(viol_dist), 2),
            "relative_heading_at_violation_deg": rel_yaw,
            "kinematics": kin,
        })
        actor_trajectories[f"V{number}"] = traj

    geometry = np.asarray(body_geometry)
    low = geometry.min(axis=0) - margin_m
    high = geometry.max(axis=0) + margin_m
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.set_xlabel("map x (m)")
    axis.set_ylabel("map y (m)")
    axis.legend(handles=handles, loc="upper right", fontsize=6.5,
                framealpha=0.9, ncol=1 if len(handles) <= 8 else 2)
    axis.set_title(title or (
        f"BEV OVERVIEW frame {start:02d}-{end:02d}: complete EGO"
        + f" and {len(nearby)} nearby vehicles + headings"))
    figure.tight_layout()

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)

    if violation_zoom_out:
        zoom_output = Path(violation_zoom_out).expanduser().resolve()
        zoom_output.parent.mkdir(parents=True, exist_ok=True)
        ego_utm = sf.ego_body(frames[violation], violation)
        ego_shape = transformed_body(ego_utm, transform)
        lane_union = layers["lanes_polygons"].geometry.unary_union
        ego_polygon = ShapelyPolygon(ego_shape)
        coverage = ego_polygon.intersection(lane_union).area / ego_polygon.area
        outside = [(number, point, lane_union.distance(Point(point)))
                   for number, point in enumerate(ego_shape[:-1])
                   if not lane_union.contains(Point(point))]
        zoom, zoom_axis = plt.subplots(figsize=(9, 8))
        layers["lanes_polygons"].plot(ax=zoom_axis, facecolor="#edf1f5",
                                      edgecolor="#8e9aaa", linewidth=0.8, zorder=0)
        layers["road_segments"].plot(ax=zoom_axis, facecolor="none",
                                      edgecolor="#596573", linewidth=1.0, zorder=1)
        layers["baseline_paths"].plot(ax=zoom_axis, color="#d5a21c",
                                       linewidth=0.7, alpha=0.8, zorder=2)
        layers["intersections"].boundary.plot(ax=zoom_axis, color="black",
                                               linewidth=1.8, zorder=3)
        for track, rows, _, _ in nearby:
            item = rows.get(violation)
            if item is None:
                continue
            actor_utm = sf.actor_body(item, frames[violation], violation)
            actor_shape = transformed_body(actor_utm, transform)
            zoom_axis.add_patch(Polygon(actor_shape, closed=True, facecolor=other_color,
                                        edgecolor="black", linewidth=0.75,
                                        alpha=0.35, zorder=6))
        center = source_body_center(ego_utm)
        _, _, yaw = heading(center[0], center[1],
                            sf.ego_heading(frames[violation], violation))
        draw_body(zoom_axis, ego_shape, yaw, ego_color, f"E{violation}",
                  alpha=0.78, zorder=8, edgecolor="#ff1744", linewidth=2.2)
        for number, point, distance in outside:
            zoom_axis.scatter([point[0]], [point[1]], s=70, facecolors="none",
                              edgecolors="#d32f2f", linewidths=2, zorder=12)
            zoom_axis.text(point[0] + 0.3, point[1] + 0.3,
                           f"C{number} outside {distance:.3f} m",
                           fontsize=8, color="#d32f2f", zorder=13)
        geometry = ego_shape
        low = geometry.min(axis=0) - violation_zoom_margin_m
        high = geometry.max(axis=0) + violation_zoom_margin_m
        zoom_axis.set_xlim(low[0], high[0])
        zoom_axis.set_ylim(low[1], high[1])
        zoom_axis.set_aspect("equal", adjustable="box")
        zoom_axis.grid(alpha=0.22)
        zoom_axis.set_xlabel("map x (m)")
        zoom_axis.set_ylabel("map y (m)")
        detail = ", ".join(f"C{number}={distance:.3f} m" for number, _, distance in outside)
        zoom_axis.set_title(
            f"Violation frame {violation:02d}: body/lane coverage={coverage:.3%}"
            + (f"; outside corners: {detail}" if detail else "; all corners inside lane"))
        zoom.tight_layout()
        zoom.savefig(zoom_output, dpi=dpi)
        plt.close(zoom)

    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "archive": str(archive),
        "token": full_token,
        "official_token": official_token,
        "outcome": outcome,
        "violation_step": violation,
        "frame_range": [start, end],
        "nearby_radius_m": nearby_radius_m,
        "ego_kinematics": ego_kinematics,
        "nearby_vehicles": actor_records,
        "trajectories": {"EGO": ego_traj, **actor_trajectories},
        "image": str(output),
        "violation_zoom": None if violation_zoom_out is None else str(Path(violation_zoom_out).expanduser().resolve()),
    }, ensure_ascii=False, indent=2))
    return {"image": str(output), "metadata": str(sidecar),
            "ego_kinematics": ego_kinematics,
            "nearby_vehicles": actor_records,
            "violation_step": violation, "frame_range": [start, end], "outcome": outcome,
            "violation_zoom": None if violation_zoom_out is None else str(Path(violation_zoom_out).expanduser().resolve())}


def score_zero_tokens(archive):
    path = Path(archive) / "scene_index.csv"
    if not path.is_file():
        raise ValueError(f"missing scene_index.csv: {path}")
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                score = float(row.get("score") or "nan")
            except ValueError:
                raise ValueError(f"invalid score for token {row.get('token')!r}") from None
            if math.isfinite(score) and score == 0.0:
                token = (row.get("token") or "").strip()
                if not token:
                    raise ValueError("score-zero scene has an empty token")
                rows.append(token)
    if not rows:
        raise ValueError(f"no score-zero scenes in {path}")
    return sorted(set(rows))


def _batch_job(arguments):
    token, archive, output, nearby_radius_m, highlight_track, margin_m, dpi, zoom_out = arguments
    if Path(output).is_file() and Path(output).with_suffix(".json").is_file():
        return {"token": token, "status": "skipped_existing", "image": output}
    try:
        result = plot_failure_bev(
            archive, token, output, nearby_radius_m=nearby_radius_m,
            highlight_track=highlight_track, margin_m=margin_m, dpi=dpi,
            violation_zoom_out=zoom_out)
        return {"token": token, "status": "success", **result}
    except Exception as error:
        return {"token": token, "status": "error", "error_type": type(error).__name__,
                "error": str(error)}


def plot_all_score_zero(archive, output_dir, *, workers=4, nearby_radius_m=30.0,
                        highlight_track=None, margin_m=8.0, dpi=150,
                        violation_zoom_dir=None):
    archive = Path(archive).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = score_zero_tokens(archive)
    zoom_dir = None if violation_zoom_dir is None else Path(violation_zoom_dir).expanduser().resolve()
    if zoom_dir is not None:
        zoom_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(token, str(archive), str(output_dir / f"{token}.png"), nearby_radius_m,
             highlight_track, margin_m, dpi,
             None if zoom_dir is None else str(zoom_dir / f"{token}.png")) for token in tokens]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(_batch_job, job): job[0] for job in jobs}
        for future in as_completed(pending):
            records.append(future.result())
    records.sort(key=lambda row: row["token"])
    success = [row for row in records if row["status"] in ("success", "skipped_existing")]
    failed = [row for row in records if row["status"] not in ("success", "skipped_existing")]
    manifest = {"archive": str(archive), "expected": len(tokens), "success": len(success),
                "failed": len(failed), "workers": workers, "records": records}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"output_dir": str(output_dir), "manifest": str(manifest_path),
            "expected": len(tokens), "success": len(success), "failed": len(failed)}


def parser():
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--archive", required=True, type=Path,
                        help="Closed-loop run directory containing scene_index.csv")
    result.add_argument("--token", help="Full scene token or unique suffix")
    result.add_argument("--out", type=Path, help="Output PNG path for one scene")
    result.add_argument("--batch-score-zero", action="store_true",
                        help="Render every score-zero scene in scene_index.csv")
    result.add_argument("--out-dir", type=Path, help="Output directory in batch mode")
    result.add_argument("--workers", type=int, default=4, help="Batch worker count (default: 4)")
    result.add_argument("--frame-range", help="Inclusive START:END; default is 0 through violation + 1")
    result.add_argument("--nearby-radius-m", type=float, default=30.0,
                        help="Include vehicles whose bodies come this close to EGO")
    result.add_argument("--highlight-track", help="Draw one selected vehicle trajectory more prominently")
    result.add_argument("--violation-zoom-out", type=Path,
                        help="Write a close-up of the violation frame and out-of-lane corners")
    result.add_argument("--violation-zoom-dir", type=Path,
                        help="Write one violation close-up per case in batch mode")
    result.add_argument("--margin-m", type=float, default=8.0, help="Margin around complete body extents")
    result.add_argument("--dpi", type=int, default=150)
    result.add_argument("--title")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.margin_m < 0:
        raise SystemExit("margin-m must be nonnegative")
    if args.nearby_radius_m < 0:
        raise SystemExit("nearby-radius-m must be nonnegative")
    if args.batch_score_zero:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --batch-score-zero")
        if args.workers < 1:
            raise SystemExit("workers must be positive")
        try:
            result = plot_all_score_zero(
                args.archive, args.out_dir, workers=args.workers,
                nearby_radius_m=args.nearby_radius_m,
                highlight_track=args.highlight_track,
                margin_m=args.margin_m, dpi=args.dpi,
                violation_zoom_dir=args.violation_zoom_dir)
        except (ValueError, OSError, sf.FactsError) as error:
            raise SystemExit(str(error)) from None
        print(json.dumps(result, ensure_ascii=False))
        return
    if not args.token or not args.out:
        raise SystemExit("--token and --out are required outside batch mode")
    try:
        result = plot_failure_bev(
            args.archive, args.token, args.out, requested_range=args.frame_range,
            nearby_radius_m=args.nearby_radius_m, highlight_track=args.highlight_track,
            margin_m=args.margin_m, dpi=args.dpi, title=args.title,
            violation_zoom_out=args.violation_zoom_out)
    except (ValueError, OSError, sf.FactsError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
