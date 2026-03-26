"""
Create an interactive map comparing predicted vs ground truth entrance locations.
"""
import json
import math
import argparse
from pathlib import Path


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True,
                        help='updated_entrances.json')
    parser.add_argument('--ground-truth', required=True,
                        help='ground_truth_labels.json')
    parser.add_argument('--buildings-json', required=True)
    parser.add_argument('--output', default='entrance_map.html')
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    pred_by_id = {p['poi_id']: p for p in predictions}

    with open(args.ground_truth) as f:
        gt_labels = json.load(f)

    with open(args.buildings_json) as f:
        buildings = json.load(f)

    # Match predictions to RTK ground truth
    matched = []
    for poi_id, gt in gt_labels.items():
        if poi_id not in pred_by_id:
            continue
        p = pred_by_id[poi_id]
        gt_lat = gt['rtk_entrance_lat']
        gt_lon = gt['rtk_entrance_lon']
        pred_lat = p['predicted_entrance_lat']
        pred_lon = p['predicted_entrance_lon']
        error = haversine_m(gt_lat, gt_lon, pred_lat, pred_lon)

        # Midpoint baseline for comparison
        bld_raw = buildings.get(p['building_id'], [])
        mid_error = None
        if bld_raw and len(bld_raw) >= 3:
            bld_lnglat = [[c[1], c[0]] for c in bld_raw]
            from update_embedding_tiles import find_facade_edge
            edge = find_facade_edge(bld_lnglat, gt_lat, gt_lon)
            if edge:
                a, b = edge['a_lnglat'], edge['b_lnglat']
                mid_lat = (a[1] + b[1]) / 2
                mid_lon = (a[0] + b[0]) / 2
                mid_error = haversine_m(gt_lat, gt_lon, mid_lat, mid_lon)

        matched.append({
            'poi_id': poi_id,
            'poi_name': p.get('poi_name', ''),
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'error_m': error,
            'mid_error_m': mid_error,
            'source': gt.get('source', 'unknown'),
        })

    # Compute stats
    errors = [m['error_m'] for m in matched]
    mid_errors = [m['mid_error_m'] for m in matched if m['mid_error_m'] is not None]
    n = len(errors)
    errors_sorted = sorted(errors)
    mid_sorted = sorted(mid_errors)

    def pct(arr, p):
        idx = int(len(arr) * p / 100)
        return arr[min(idx, len(arr) - 1)]

    stats = {
        'n': n,
        'mae': sum(errors) / n,
        'median': errors_sorted[n // 2],
        'p90': pct(errors_sorted, 90),
        'lt1': sum(1 for e in errors if e < 1),
        'lt2': sum(1 for e in errors if e < 2),
        'lt5': sum(1 for e in errors if e < 5),
        'mid_mae': sum(mid_errors) / len(mid_errors) if mid_errors else 0,
        'mid_median': mid_sorted[len(mid_sorted) // 2] if mid_sorted else 0,
        'mid_p90': pct(mid_sorted, 90) if mid_sorted else 0,
    }

    print(f"Matched {n} RTK POIs")
    print(f"Prediction: MAE={stats['mae']:.2f}m, Median={stats['median']:.2f}m, P90={stats['p90']:.2f}m")
    print(f"Midpoint:   MAE={stats['mid_mae']:.2f}m, Median={stats['mid_median']:.2f}m, P90={stats['mid_p90']:.2f}m")

    # Center map
    center_lat = sum(m['gt_lat'] for m in matched) / n
    center_lon = sum(m['gt_lon'] for m in matched) / n

    # Build markers JS
    markers_js = []
    for m in matched:
        color = 'green' if m['error_m'] < 2 else ('orange' if m['error_m'] < 5 else 'red')
        mid_str = f", midpoint: {m['mid_error_m']:.1f}m" if m['mid_error_m'] else ""
        name = m['poi_name'].replace("'", "\\'")
        markers_js.append(f"""
        // GT marker
        L.circleMarker([{m['gt_lat']}, {m['gt_lon']}], {{
            radius: 6, color: '#333', fillColor: '#2196F3', fillOpacity: 0.9, weight: 1
        }}).addTo(map).bindPopup('<b>{name}</b><br>Ground Truth<br>Source: {m["source"]}')
          .bindTooltip('{name}', {{permanent: true, direction: 'right', offset: [8, 0], className: 'poi-label'}});
        // Pred marker
        L.circleMarker([{m['pred_lat']}, {m['pred_lon']}], {{
            radius: 6, color: '#333', fillColor: '{color}', fillOpacity: 0.9, weight: 1
        }}).addTo(map).bindPopup('<b>{name}</b><br>Prediction<br>Error: {m["error_m"]:.1f}m{mid_str}');
        // Line
        L.polyline([[{m['gt_lat']}, {m['gt_lon']}], [{m['pred_lat']}, {m['pred_lon']}]], {{
            color: '{color}', weight: 2, opacity: 0.7
        }}).addTo(map);""")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Entrance Prediction Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        .stats-panel {{
            position: absolute; top: 10px; right: 10px; z-index: 1000;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: sans-serif;
            font-size: 13px; max-width: 320px;
        }}
        .stats-panel h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
        .stats-panel table {{ border-collapse: collapse; width: 100%; }}
        .stats-panel td {{ padding: 2px 8px; }}
        .stats-panel .header {{ font-weight: bold; border-bottom: 1px solid #ccc; }}
        .legend {{ margin-top: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; border: 1px solid #333; }}
        .poi-label {{
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 11px;
            font-weight: 600;
            font-family: sans-serif;
            color: #333;
            text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;
            white-space: nowrap;
        }}
        .poi-label::before {{ display: none; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="stats-panel">
        <h3>Entrance Prediction vs RTK Ground Truth</h3>
        <table>
            <tr class="header"><td></td><td>Prediction</td><td>Midpoint</td></tr>
            <tr><td>MAE</td><td>{stats['mae']:.2f}m</td><td>{stats['mid_mae']:.2f}m</td></tr>
            <tr><td>Median</td><td>{stats['median']:.2f}m</td><td>{stats['mid_median']:.2f}m</td></tr>
            <tr><td>P90</td><td>{stats['p90']:.2f}m</td><td>{stats['mid_p90']:.2f}m</td></tr>
            <tr><td>&lt;1m</td><td>{stats['lt1']/n*100:.0f}%</td><td></td></tr>
            <tr><td>&lt;2m</td><td>{stats['lt2']/n*100:.0f}%</td><td></td></tr>
            <tr><td>&lt;5m</td><td>{stats['lt5']/n*100:.0f}%</td><td></td></tr>
        </table>
        <div class="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#2196F3"></div>Ground Truth (RTK)</div>
            <div class="legend-item"><div class="legend-dot" style="background:green"></div>Prediction &lt;2m</div>
            <div class="legend-item"><div class="legend-dot" style="background:orange"></div>Prediction 2-5m</div>
            <div class="legend-item"><div class="legend-dot" style="background:red"></div>Prediction &gt;5m</div>
        </div>
        <div style="margin-top:8px;color:#666;font-size:11px">{n} RTK-surveyed POIs</div>
    </div>
    <script>
        var map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}@2x.png', {{
            attribution: '&copy; OpenStreetMap, &copy; CARTO',
            maxZoom: 20
        }}).addTo(map);
        {''.join(markers_js)}

        // Hide overlapping labels
        function hideOverlaps() {{
            var labels = document.querySelectorAll('.poi-label');
            var rects = [];
            labels.forEach(function(el) {{
                el.style.display = '';
                var r = el.getBoundingClientRect();
                rects.push({{el: el, left: r.left, top: r.top, right: r.right, bottom: r.bottom}});
            }});
            for (var i = 0; i < rects.length; i++) {{
                if (rects[i].el.style.display === 'none') continue;
                for (var j = i + 1; j < rects.length; j++) {{
                    if (rects[j].el.style.display === 'none') continue;
                    var a = rects[i], b = rects[j];
                    if (a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top) {{
                        rects[j].el.style.display = 'none';
                    }}
                }}
            }}
        }}
        map.on('zoomend moveend', hideOverlaps);
        setTimeout(hideOverlaps, 500);
    </script>
</body>
</html>"""

    with open(args.output, 'w') as f:
        f.write(html)
    print(f"Saved map to {args.output}")


if __name__ == '__main__':
    main()
