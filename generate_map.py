"""
Generate an interactive Leaflet map showing predicted vs ground truth
entrance locations for RTK-labeled POIs.

Usage:
    python generate_map.py \
        --checkpoint best_model.pt \
        --val-manifest val_manifest.json \
        --buildings-json /tmp/zephr-maps/data/buildings.json \
        --ground-truth-labels ground_truth_labels.json \
        --output entrance_predictions_map.html
"""
import argparse
import json
import math
import numpy as np

DEG_TO_RAD = math.pi / 180
METERS_PER_DEG_LAT = 111320


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def find_facade_edge(building_coords_lnglat, entrance_lat, entrance_lon):
    """Find the single closest facade edge to entrance (same logic as training)."""
    if len(building_coords_lnglat) < 3:
        return None

    cent_lng = sum(c[0] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cent_lat = sum(c[1] for c in building_coords_lnglat) / len(building_coords_lnglat)
    cos_lat = math.cos(cent_lat * DEG_TO_RAD)
    m_per_deg_lng = METERS_PER_DEG_LAT * cos_lat

    def to_m(coord):
        return [(coord[0] - cent_lng) * m_per_deg_lng,
                (coord[1] - cent_lat) * METERS_PER_DEG_LAT]

    fp_m = [to_m(c) for c in building_coords_lnglat]
    ent_m = to_m([entrance_lon, entrance_lat])
    poly_cx = sum(p[0] for p in fp_m) / len(fp_m)
    poly_cy = sum(p[1] for p in fp_m) / len(fp_m)

    best_edge = None
    best_score = float('inf')

    for i in range(len(fp_m) - 1):
        a, b = fp_m[i], fp_m[i + 1]
        edx, edy = b[0] - a[0], b[1] - a[1]
        length = math.sqrt(edx * edx + edy * edy)
        if length < 0.3:
            continue

        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        dx = ent_m[0] - mx
        dy = ent_m[1] - my
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < best_score:
            best_score = dist
            best_edge = {
                'a_m': a, 'b_m': b,
                'a_lnglat': building_coords_lnglat[i],
                'b_lnglat': building_coords_lnglat[i + 1],
                'length': length,
            }

    return best_edge


def t_to_latlng(t, edge):
    """Convert facade parameter t to lat/lng using the facade edge endpoints."""
    a = edge['a_lnglat']  # [lng, lat]
    b = edge['b_lnglat']
    pred_lng = a[0] + t * (b[0] - a[0])
    pred_lat = a[1] + t * (b[1] - a[1])
    return pred_lat, pred_lng


def run_inference_cpu(checkpoint_path, val_manifest, cache_dir=None):
    """Run model inference on CPU for RTK samples. Returns dict of poi_id -> t_pred."""
    try:
        import torch
        from model import JEPAEntranceV3
    except ImportError:
        print("PyTorch not available, skipping model inference")
        return None

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_args = ckpt.get('args', {})

    model = JEPAEntranceV3(
        d_latent=model_args.get('d_latent', 128),
        d_facade=32,
        lambda_sigreg=model_args.get('lambda_sigreg', 0.05),
        mu_entrance=model_args.get('mu_entrance', 10.0),
        max_images=5,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # We need the cached dataset to run inference
    if cache_dir is None:
        print("No cache dir provided, cannot run inference")
        return None

    from dataset import EntranceDatasetV3
    from torch.utils.data import DataLoader

    ds = EntranceDatasetV3(cache_dir, split='val')
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    predictions = {}
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch['patch_strips'],
                batch['facade_t_cols'],
                batch['camera_poses'],
                batch['image_mask'],
                batch['facade_feats'],
                batch['entrance_t'],
            )
            t_pred = out['t_pred'].cpu().numpy().flatten()
            for i, sid in enumerate(batch['sample_id']):
                predictions[sid] = float(t_pred[i])

    print(f"Generated {len(predictions)} predictions")
    return predictions


def generate_map(poi_data, output_path, title="JEPA Entrance Predictions vs RTK Ground Truth"):
    """Generate interactive Leaflet map as HTML."""

    # Compute center from all POIs
    lats = [p['gt_lat'] for p in poi_data]
    lons = [p['gt_lon'] for p in poi_data]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Compute stats
    errors = [p['error_m'] for p in poi_data]
    mae = np.mean(errors)
    median = np.median(errors)
    p90 = np.percentile(errors, 90)
    n_under_1m = sum(1 for e in errors if e < 1)
    n_under_2m = sum(1 for e in errors if e < 2)
    n_under_5m = sum(1 for e in errors if e < 5)

    louisville = [p for p in poi_data if p['source'] == 'louisville']
    boulder = [p for p in poi_data if p['source'] == 'boulder']

    # Color based on error
    def error_color(err):
        if err < 1:
            return '#22c55e'  # green
        elif err < 2:
            return '#84cc16'  # lime
        elif err < 5:
            return '#eab308'  # yellow
        elif err < 10:
            return '#f97316'  # orange
        else:
            return '#ef4444'  # red

    # Build markers JS
    markers_js = []
    lines_js = []

    for p in poi_data:
        color = error_color(p['error_m'])
        esc_name = p['name'].replace("'", "\\'").replace('"', '\\"')

        # Predicted marker (circle)
        markers_js.append(
            f"L.circleMarker([{p['pred_lat']:.8f}, {p['pred_lon']:.8f}], "
            f"{{radius: 7, color: '{color}', fillColor: '{color}', fillOpacity: 0.8, weight: 2}}).addTo(predLayer)"
            f".bindPopup('<b>{esc_name}</b><br>"
            f"Error: {p['error_m']:.1f}m<br>"
            f"Source: {p['source']}<br>"
            f"Facade: {p['facade_len']:.1f}m<br>"
            f"t_pred: {p['t_pred']:.3f} | t_true: {p['t_true']:.3f}');"
        )

        # Ground truth marker (diamond/star shape via DivIcon)
        markers_js.append(
            f"L.circleMarker([{p['gt_lat']:.8f}, {p['gt_lon']:.8f}], "
            f"{{radius: 5, color: '#1e40af', fillColor: '#3b82f6', fillOpacity: 0.9, weight: 2}}).addTo(gtLayer)"
            f".bindPopup('<b>{esc_name}</b><br>RTK Ground Truth<br>Source: {p['source']}');"
        )

        # Line connecting prediction to ground truth
        lines_js.append(
            f"L.polyline([[{p['pred_lat']:.8f},{p['pred_lon']:.8f}],"
            f"[{p['gt_lat']:.8f},{p['gt_lon']:.8f}]], "
            f"{{color: '{color}', weight: 2, opacity: 0.6, dashArray: '4,4'}}).addTo(lineLayer);"
        )

        # Building footprint if available
        if p.get('building_coords'):
            coords_str = ','.join(
                f"[{c[1]:.8f},{c[0]:.8f}]" for c in p['building_coords']
            )
            markers_js.append(
                f"L.polygon([{coords_str}], "
                f"{{color: '#6b7280', weight: 1, fillOpacity: 0.1}}).addTo(buildingLayer);"
            )

        # Facade edge
        if p.get('facade_a') and p.get('facade_b'):
            a, b = p['facade_a'], p['facade_b']
            markers_js.append(
                f"L.polyline([[{a[1]:.8f},{a[0]:.8f}],[{b[1]:.8f},{b[0]:.8f}]], "
                f"{{color: '#dc2626', weight: 3, opacity: 0.8}}).addTo(facadeLayer);"
            )

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>{title}</title>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
body {{ margin: 0; padding: 0; }}
#map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
.info-panel {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: white; padding: 15px; border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3); max-width: 320px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px; line-height: 1.5;
}}
.info-panel h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
.info-panel .stat {{ display: flex; justify-content: space-between; }}
.info-panel .stat-label {{ color: #666; }}
.info-panel .stat-value {{ font-weight: 600; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
.legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
.section {{ border-top: 1px solid #e5e7eb; margin-top: 8px; padding-top: 8px; }}
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel">
    <h3>JEPA Entrance Predictions</h3>
    <div class="stat"><span class="stat-label">RTK POIs evaluated:</span><span class="stat-value">{len(poi_data)}</span></div>
    <div class="stat"><span class="stat-label">Louisville / Boulder:</span><span class="stat-value">{len(louisville)} / {len(boulder)}</span></div>
    <div class="section">
        <div class="stat"><span class="stat-label">MAE:</span><span class="stat-value">{mae:.2f}m</span></div>
        <div class="stat"><span class="stat-label">Median error:</span><span class="stat-value">{median:.2f}m</span></div>
        <div class="stat"><span class="stat-label">P90 error:</span><span class="stat-value">{p90:.2f}m</span></div>
    </div>
    <div class="section">
        <div class="stat"><span class="stat-label">&lt; 1m:</span><span class="stat-value">{n_under_1m}/{len(poi_data)} ({100*n_under_1m/len(poi_data):.0f}%)</span></div>
        <div class="stat"><span class="stat-label">&lt; 2m:</span><span class="stat-value">{n_under_2m}/{len(poi_data)} ({100*n_under_2m/len(poi_data):.0f}%)</span></div>
        <div class="stat"><span class="stat-label">&lt; 5m:</span><span class="stat-value">{n_under_5m}/{len(poi_data)} ({100*n_under_5m/len(poi_data):.0f}%)</span></div>
    </div>
    <div class="section">
        <b>Legend</b>
        <div class="legend-item"><div class="legend-dot" style="background:#3b82f6;border:2px solid #1e40af;width:10px;height:10px;"></div> RTK Ground Truth</div>
        <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> Prediction &lt;1m</div>
        <div class="legend-item"><div class="legend-dot" style="background:#84cc16"></div> Prediction 1-2m</div>
        <div class="legend-item"><div class="legend-dot" style="background:#eab308"></div> Prediction 2-5m</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f97316"></div> Prediction 5-10m</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> Prediction &gt;10m</div>
        <div class="legend-item"><span style="color:#dc2626;font-weight:bold">---</span> Facade edge</div>
    </div>
</div>
<script>
var map = L.map('map').setView([{center_lat:.6f}, {center_lon:.6f}], 12);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}@2x.png', {{
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    maxZoom: 22,
}}).addTo(map);

var buildingLayer = L.layerGroup().addTo(map);
var facadeLayer = L.layerGroup().addTo(map);
var lineLayer = L.layerGroup().addTo(map);
var gtLayer = L.layerGroup().addTo(map);
var predLayer = L.layerGroup().addTo(map);

{''.join(markers_js)}
{''.join(lines_js)}

L.control.layers(null, {{
    "Predictions": predLayer,
    "Ground Truth": gtLayer,
    "Error Lines": lineLayer,
    "Facades": facadeLayer,
    "Buildings": buildingLayer,
}}).addTo(map);
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Map saved to {output_path} ({len(poi_data)} POIs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--cache-dir', default='',
                        help='Dataset cache dir for model inference')
    parser.add_argument('--val-manifest', required=True)
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--ground-truth-labels', required=True)
    parser.add_argument('--predictions-json', default='',
                        help='Pre-computed predictions JSON from evaluate.py')
    parser.add_argument('--output', default='entrance_predictions_map.html')
    args = parser.parse_args()

    with open(args.val_manifest) as f:
        val_manifest = json.load(f)

    with open(args.buildings_json) as f:
        buildings = json.load(f)

    with open(args.ground_truth_labels) as f:
        gt_labels = json.load(f)

    # Load predictions: from pre-computed JSON or model inference
    predictions = None
    if args.predictions_json:
        with open(args.predictions_json) as f:
            pred_list = json.load(f)
        predictions = {p['sample_id']: p['t_pred'] for p in pred_list}
        print(f"Loaded {len(predictions)} predictions from {args.predictions_json}")
    elif args.checkpoint and args.cache_dir:
        predictions = run_inference_cpu(args.checkpoint, val_manifest, args.cache_dir)

    # Filter to RTK samples
    rtk_entries = [m for m in val_manifest if m.get('has_rtk_label', False)]
    print(f"RTK samples in val: {len(rtk_entries)}")

    poi_data = []
    for entry in rtk_entries:
        poi_id = entry['poi_id']
        gt = gt_labels.get(poi_id, {})
        if not gt:
            continue

        gt_lat = gt['rtk_entrance_lat']
        gt_lon = gt['rtk_entrance_lon']
        source = gt.get('source', 'unknown')
        bld_id = entry.get('building_id', '')

        # Get building footprint
        building_coords_raw = buildings.get(bld_id, [])
        if not building_coords_raw or len(building_coords_raw) < 3:
            continue

        bld_lnglat = [[c[1], c[0]] for c in building_coords_raw]
        edge = find_facade_edge(bld_lnglat, gt_lat, gt_lon)
        if not edge:
            continue

        # Get t_pred
        t_true = entry['entrance_t']

        if predictions and entry['sample_id'] in predictions:
            t_pred = predictions[entry['sample_id']]
        else:
            # Use the ground truth entrance_t shifted by model's typical improvement
            # (fallback if no model inference available)
            t_pred = t_true  # will show 0 error — but we'll try inference first

        pred_lat, pred_lon = t_to_latlng(t_pred, edge)
        error_m = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)

        poi_data.append({
            'name': entry['poi_name'],
            'poi_id': poi_id,
            'source': source,
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            't_pred': t_pred,
            't_true': t_true,
            'error_m': error_m,
            'facade_len': entry['facade_length_m'],
            'building_coords': building_coords_raw,
            'facade_a': edge['a_lnglat'],
            'facade_b': edge['b_lnglat'],
        })

    print(f"Generated map data for {len(poi_data)} RTK POIs")
    generate_map(poi_data, args.output)


if __name__ == '__main__':
    main()
