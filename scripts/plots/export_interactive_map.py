#!/usr/bin/env python3
"""Export a standalone Leaflet report from stack COGs and site GeoJSON/JSON data."""

from __future__ import annotations

import argparse
import base64
import html as html_utils
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any


PALETTE = {
    "metro": "#e05b51",
    "rail": "#f5a742",
    "light_rail": "#58a6d8",
    "motorway": "#ef8f36",
    "trunk": "#d68843",
    "canal": "#42b5a0",
    "river": "#39a7e0",
    "reservoir": "#a78bda",
    "dam": "#c084fc",
    "airport_runway": "#d7dde7",
    "infrastructure": "#aab5c4",
}


def _json_file(path: Path | None, fallback: Any) -> Any:
    if path is None or not path.is_file():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_json(value: Any) -> str:
    """Serialize JSON safely inside a script element."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _date_range(corridor_history: dict[str, Any]) -> tuple[date | None, date | None, list[str]]:
    epochs = [str(value) for value in corridor_history.get("epochs", []) if value]
    parsed: list[date] = []
    for value in epochs:
        try:
            parsed.append(date.fromisoformat(value[:10]))
        except ValueError:
            continue
    if parsed:
        return min(parsed), max(parsed), epochs
    try:
        origin = date.fromisoformat(str(corridor_history.get("origin_date", ""))[:10])
        return origin, origin, [origin.isoformat()]
    except ValueError:
        return None, None, []


def _read_unr_series(
    station: dict[str, Any],
    gnss_cache_dir: Path | None,
    start: date | None,
    end: date | None,
) -> dict[str, Any] | None:
    if gnss_cache_dir is None or start is None or end is None:
        return None
    station_id = str(station.get("station_id", ""))
    path = gnss_cache_dir / f"{station_id}.NA.tenv3"
    if not path.is_file():
        return None

    mjd_origin = date(1858, 11, 17)
    samples: list[tuple[str, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 23 or fields[0].lower() == "site":
            continue
        try:
            observed = mjd_origin + timedelta(days=int(fields[3]))
            if observed < start or observed > end:
                continue
            east = float(fields[7]) + float(fields[8])
            north = float(fields[9]) + float(fields[10])
            up = float(fields[11]) + float(fields[12])
            if not all(math.isfinite(v) for v in (east, north, up)):
                continue
            samples.append((observed.isoformat(), east, north, up))
        except (ValueError, OverflowError):
            continue

    if len(samples) < 2:
        return None

    east0, north0, up0 = samples[0][1:]
    los_e = float(station.get("los_e") or 0.0)
    los_n = float(station.get("los_n") or 0.0)
    los_u = float(station.get("los_u") or 0.0)
    values = [
        round(
            1000.0
            * (
                los_e * (east - east0)
                + los_n * (north - north0)
                + los_u * (up - up0)
            ),
            3,
        )
        for _, east, north, up in samples
    ]
    return {
        "dates": [sample[0] for sample in samples],
        "displacement_mm": values,
        "label": "UNR daily GNSS LOS displacement",
        "source": "Observed daily UNR .tenv3 positions projected onto the station LOS vector",
        "is_estimate": False,
    }


def _station_features(
    validation: dict[str, Any],
    gnss_cache_dir: Path | None,
    corridor_history: dict[str, Any],
) -> dict[str, Any]:
    start, end, epochs = _date_range(corridor_history)
    origin = start or date.today()
    features: list[dict[str, Any]] = []
    for station in validation.get("stations", []):
        try:
            lon, lat = float(station["lon"]), float(station["lat"])
        except (KeyError, TypeError, ValueError):
            continue

        history = _read_unr_series(station, gnss_cache_dir, start, end)
        if history is None:
            dates = epochs or [origin.isoformat()]
            rate = float(station.get("gnss_los_mm_yr") or 0.0)
            history = {
                "dates": dates,
                "displacement_mm": [
                    round((date.fromisoformat(value[:10]) - origin).days * rate / 365.25, 3)
                    for value in dates
                ],
                "label": "GNSS LOS displacement estimate",
                "source": "Linear estimate from the validation GNSS LOS velocity over the stack dates",
                "is_estimate": True,
            }

        insar_rate = station.get("insar_los_mm_yr")
        if insar_rate is not None:
            history["insar_displacement_mm"] = [
                round((date.fromisoformat(value[:10]) - origin).days * float(insar_rate) / 365.25, 3)
                for value in history["dates"]
            ]
            history["insar_label"] = "InSAR velocity trend (linearized)"

        props = {
            "station_id": station.get("station_id", "GNSS station"),
            "height_m": station.get("height_m"),
            "gnss_los_mm_yr": station.get("gnss_los_mm_yr"),
            "insar_los_mm_yr": station.get("insar_los_mm_yr"),
            "insar_coherence": station.get("insar_coherence"),
            "residual_mm_yr": station.get("residual_mm_yr"),
            "timeseries": history,
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _raster_base64(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Raster not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_report(
    *,
    site_name: str,
    stack_dir: Path,
    corridors_path: Path,
    validation_path: Path | None,
    gnss_cache_dir: Path | None,
    output_path: Path,
    primary_layer: str,
    compare_layer: str,
    primary_label: str,
    compare_label: str,
) -> Path:
    corridors = _json_file(corridors_path, {"type": "FeatureCollection", "features": []})
    validation = _json_file(validation_path, {})
    summary = _json_file(stack_dir / "stack_summary.json", {})
    corridor_history = _json_file(stack_dir / "corridor_timeseries.json", {})
    history_by_id = corridor_history.get("corridors", {}) if isinstance(corridor_history, dict) else {}
    stations = _station_features(validation, gnss_cache_dir, corridor_history)

    raster_data = {
        "primary": _raster_base64(stack_dir / primary_layer),
        "compare": _raster_base64(stack_dir / compare_layer),
        "primaryName": primary_layer,
        "compareName": compare_layer,
        "primaryLabel": primary_label,
        "compareLabel": compare_label,
    }
    report_data = {
        "siteName": site_name,
        "rasters": raster_data,
        "corridors": corridors,
        "corridorHistory": history_by_id,
        "stations": stations,
        "summary": summary,
        "gnssValidation": validation.get("metrics", {}),
    }

    safe_site_name = html_utils.escape(site_name)
    colors = _safe_json(PALETTE)
    payload = _safe_json(report_data)
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_site_name} · InSAR interactive report</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
  <style>
    :root {{ color-scheme: dark; --bg:#10151c; --panel:#171e27; --line:#2a3542; --text:#edf2f7; --muted:#a9b6c5; --accent:#69d3be; }}
    * {{ box-sizing:border-box; }} html,body {{ height:100%; margin:0; background:var(--bg); color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
    .shell {{ height:100%; display:grid; grid-template-rows:auto auto minmax(0,1fr); }}
    header {{ padding:15px 20px 11px; display:flex; align-items:flex-end; justify-content:space-between; gap:18px; border-bottom:1px solid var(--line); background:#121820; }}
    h1 {{ margin:0; font-size:19px; letter-spacing:-.02em; }} .sub {{ color:var(--muted); font-size:12px; margin-top:3px; }}
    .metrics {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:12px; }} .metrics strong {{ color:var(--text); display:block; font-size:15px; }}
    .toolbar {{ min-height:46px; display:flex; align-items:center; gap:22px; padding:8px 18px; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }}
    .toolbar label {{ display:flex; align-items:center; gap:9px; color:var(--muted); }} .toolbar input[type=range] {{ accent-color:var(--accent); width:150px; }}
    .tag {{ color:var(--muted); font-size:12px; }} .tag b {{ color:var(--text); font-weight:600; }} #map {{ min-height:0; width:100%; background:#111821; }}
    .legend {{ padding:8px 11px; background:rgba(17,24,32,.9); border:1px solid var(--line); border-radius:8px; color:var(--muted); }} .legend strong {{ display:block; color:var(--text); margin-bottom:4px; }} .gradient {{ height:9px; width:180px; border-radius:8px; background:linear-gradient(90deg,#153e74,#64a8c7,#e9e8df,#e99c73,#8b2338); }}
    .leaflet-popup-content-wrapper,.leaflet-popup-tip {{ background:#171e27; color:var(--text); }} .leaflet-popup-content {{ min-width:270px; margin:13px 15px; }} .popup-title {{ font-weight:700; font-size:14px; margin-bottom:4px; }} .popup-meta {{ color:var(--muted); font-size:11px; margin-bottom:8px; }} .popup-chart {{ width:310px; height:150px; }}
    .leaflet-control-layers {{ background:#171e27; color:var(--text); border:1px solid var(--line); }} .leaflet-control-attribution {{ background:rgba(17,24,32,.7)!important; color:#b9c3d0!important; }}
    @media(max-width:760px) {{ header {{ align-items:flex-start; flex-direction:column; }} .metrics {{ gap:12px; }} .toolbar {{ gap:12px; }} .toolbar input[type=range] {{ width:110px; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <header><div><h1>{safe_site_name}</h1><div class="sub">InSAR stack explorer · COG detail with corridor and GNSS context</div></div><div class="metrics" id="metrics"></div></header>
    <div class="toolbar">
      <label>Raster opacity <input id="opacity" type="range" min="10" max="100" value="82"><span id="opacityValue">82%</span></label>
      <span class="tag">Swipe to compare <b>{primary_label}</b> / <b>{compare_label}</b></span>
      <span class="tag">Click a corridor or station for its displacement history</span>
    </div>
    <div id="map" role="application" aria-label="Interactive InSAR map"></div>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <script src="https://unpkg.com/proj4@2.9.0/dist/proj4.js"></script>
  <script src="https://unpkg.com/georaster@1.6.0/dist/georaster.browser.bundle.min.js"></script>
  <script src="https://unpkg.com/georaster-layer-for-leaflet@3.10.0/dist/georaster-layer-for-leaflet.min.js"></script>
  <script src="https://unpkg.com/leaflet-side-by-side@2.2.0/leaflet-side-by-side.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <script>
    const REPORT = {payload};
    const CLASS_COLORS = {colors};
    for(let zone=1;zone<=60;zone++){{
      proj4.defs(`EPSG:${{32600+zone}}`,`+proj=utm +zone=${{zone}} +datum=WGS84 +units=m +no_defs +type=crs`);
      proj4.defs(`EPSG:${{32700+zone}}`,`+proj=utm +zone=${{zone}} +south +datum=WGS84 +units=m +no_defs +type=crs`);
    }}
    const map = L.map('map', {{ zoomControl:true, preferCanvas:true }});
    const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{ maxZoom:19, attribution:'Tiles © Esri — Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community' }});
    const dark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ subdomains:'abcd', maxZoom:20, attribution:'© OpenStreetMap contributors © CARTO' }});
    satellite.addTo(map);
    L.control.layers({{ 'Satellite':satellite, 'Dark':dark }}, null, {{ collapsed:false }}).addTo(map);
    map.setView([0,0],2);

    const htmlEscape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    const fmt = (value, digits=1) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
    const metrics = REPORT.summary || {{}};
    const metricItems = [
      ['Pairs', metrics.pair_count], ['Stack span', metrics.calendar_span_days == null ? null : `${{metrics.calendar_span_days}} days`],
      ['Valid pixels', metrics.valid_stack_pixels], ['p95 |velocity|', metrics.p95_abs_velocity_mm_yr == null ? null : `${{fmt(metrics.p95_abs_velocity_mm_yr)}} mm/yr`],
      ['GNSS RMSE', REPORT.gnssValidation.rmse_mm_yr == null ? null : `${{fmt(REPORT.gnssValidation.rmse_mm_yr)}} mm/yr`]
    ].filter(item => item[1] !== undefined && item[1] !== null);
    document.getElementById('metrics').innerHTML = metricItems.map(([label,value]) => `<span>${{htmlEscape(label)}}<strong>${{htmlEscape(value)}}</strong></span>`).join('');

    const velocityStops = [[0,'#153e74'],[.24,'#64a8c7'],[.5,'#e9e8df'],[.76,'#e99c73'],[1,'#8b2338']];
    const coherenceStops = [[0,'#18202b'],[.3,'#3c5366'],[.6,'#6f949c'],[.85,'#b7c9bd'],[1,'#f5f2e6']];
    function hexRgb(hex) {{ const n=parseInt(hex.slice(1),16); return [(n>>16)&255,(n>>8)&255,n&255]; }}
    function ramp(value, stops) {{
      const x=Math.max(0,Math.min(1,value)); let i=1; while(i<stops.length && x>stops[i][0]) i++;
      const a=stops[i-1], b=stops[Math.min(i,stops.length-1)], t=(x-a[0])/Math.max(1e-9,b[0]-a[0]);
      const c1=hexRgb(a[1]), c2=hexRgb(b[1]); return `rgb(${{c1.map((v,j)=>Math.round(v+(c2[j]-v)*t)).join(',')}})`;
    }}
    function rasterStats(georaster, style) {{
      if (style === 'coherence') return 1;
      const min=Number(georaster.mins?.[0] ?? georaster.min ?? -25), max=Number(georaster.maxs?.[0] ?? georaster.max ?? 25);
      return Math.max(10, Math.abs(min), Math.abs(max));
    }}
    function colorFn(style, limit, nodata) {{ return values => {{
      const value=Number(values?.[0]);
      if (!Number.isFinite(value) || (nodata != null && value === nodata)) return 'rgba(0,0,0,0)';
      return style === 'coherence' ? ramp(value,coherenceStops) : ramp((value+limit)/(2*limit),velocityStops);
    }}; }}
    function bytesFromBase64(data) {{
      const binary=atob(data), bytes=new Uint8Array(binary.length);
      for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
      return bytes.buffer;
    }}
    const overlayLayers=[];
    async function addRaster(encoded, style, label) {{
      const georaster=await parseGeoraster(bytesFromBase64(encoded));
      const limit=rasterStats(georaster,style);
      const layer=new GeoRasterLayer({{ georaster, opacity:.82, resolution:256, pixelValuesToColorFn:colorFn(style,limit,georaster.noDataValue) }});
      layer.addTo(map); overlayLayers.push(layer);
      return layer;
    }}
    const opacity=document.getElementById('opacity');
    opacity.addEventListener('input',()=>{{ const value=Number(opacity.value)/100; overlayLayers.forEach(layer=>layer.setOpacity(value)); document.getElementById('opacityValue').textContent=`${{opacity.value}}%`; }});

    function makeChartPopup(title, meta, series) {{
      const wrap=document.createElement('div');
      const heading=document.createElement('div'); heading.className='popup-title'; heading.textContent=title; wrap.appendChild(heading);
      const sub=document.createElement('div'); sub.className='popup-meta'; sub.textContent=meta; wrap.appendChild(sub);
      const canvas=document.createElement('canvas'); canvas.className='popup-chart'; wrap.appendChild(canvas);
      const chart=new Chart(canvas, {{ type:'line', data:{{ labels:series.labels, datasets:series.datasets }}, options:{{ responsive:true, maintainAspectRatio:false, interaction:{{mode:'index',intersect:false}}, plugins:{{legend:{{labels:{{color:'#dbe4ee',boxWidth:12,font:{{size:10}}}}}}, tooltip:{{callbacks:{{label:ctx=>`${{ctx.dataset.label}}: ${{fmt(ctx.parsed.y,2)}} mm`}}}}}}, scales:{{x:{{ticks:{{color:'#a9b6c5',maxTicksLimit:6,font:{{size:9}}}},grid:{{display:false}}}},y:{{title:{{display:true,text:'Displacement (mm)',color:'#a9b6c5'}},ticks:{{color:'#a9b6c5',font:{{size:9}}}},grid:{{color:'rgba(169,182,197,.13)'}}}}}} }} }});
      return wrap;
    }}
    map.on('popupclose',event=>{{
      const content=event.popup.getContent();
      if(content instanceof HTMLElement){{const canvas=content.querySelector('canvas');if(canvas)Chart.getChart(canvas)?.destroy();}}
    }});
    function lineDataset(label, values, color, dashed=false) {{ return {{label,data:values,borderColor:color,backgroundColor:color,pointRadius:0,pointHoverRadius:3,borderWidth:2,tension:.16,spanGaps:true,borderDash:dashed?[5,4]:[]}}; }}
    function corridorPopup(feature) {{
      const props=feature.properties||{{}}, id=props.asset_id||props.id||'', history=REPORT.corridorHistory[id];
      if(!history) {{ const box=document.createElement('div'); box.className='popup-title'; box.textContent=props.name||'Corridor'; return box; }}
      const points=(history.points||[]).filter(p=>p.date && p.cumulative_disp_mm!=null);
      const values=points.map(p=>Number(p.cumulative_disp_mm));
      const latest=points[points.length-1]||{{}};
      const rate=latest.mean_vel_mm_yr==null?'':` · latest mean ${{fmt(latest.mean_vel_mm_yr)}} mm/yr`;
      return makeChartPopup(props.name||history.name||'Corridor',`${{props.asset_class||history.asset_class||'asset'}}${{rate}}`,{{labels:points.map(p=>p.date),datasets:[lineDataset('Cumulative displacement',values,'#69d3be')]}});
    }}
    const corridorLayer=L.geoJSON(REPORT.corridors,{{
      style:feature=>{{const p=feature.properties||{{}};return {{color:CLASS_COLORS[p.asset_class]||'#f5a742',weight:2.5,opacity:.88}};}},
      onEachFeature:(feature,layer)=>layer.on('click',()=>layer.bindPopup(corridorPopup(feature),{{maxWidth:370}}).openPopup())
    }}).addTo(map);
    function stationPopup(feature) {{
      const p=feature.properties||{{}}, ts=p.timeseries||{{}}, labels=ts.dates||[], datasets=[];
      if(ts.displacement_mm?.length) datasets.push(lineDataset(ts.label||'GNSS displacement',ts.displacement_mm,ts.is_estimate?'#e8ab56':'#55b8e7',ts.is_estimate));
      if(ts.insar_displacement_mm?.length) datasets.push(lineDataset(ts.insar_label||'InSAR trend',ts.insar_displacement_mm,'#ed766f',true));
      const meta=`GNSS ${{fmt(p.gnss_los_mm_yr)}} mm/yr · InSAR ${{fmt(p.insar_los_mm_yr)}} mm/yr · coherence ${{fmt(p.insar_coherence,2)}}`;
      return makeChartPopup(p.station_id||'GNSS station',`${{meta}} · ${{ts.source||'Station time series'}}`,{{labels,datasets}});
    }}
    const stationLayer=L.geoJSON(REPORT.stations,{{
      pointToLayer:(feature,latlng)=>L.circleMarker(latlng,{{radius:6,color:'#121820',weight:2,fillColor:'#ffe07b',fillOpacity:1}}),
      onEachFeature:(feature,layer)=>layer.on('click',()=>layer.bindPopup(stationPopup(feature),{{maxWidth:370}}).openPopup())
    }}).addTo(map);
    L.control.layers(null,{{'Infrastructure corridors':corridorLayer,'GNSS stations':stationLayer}},{{collapsed:true}}).addTo(map);
    const legend=L.control({{position:'bottomleft'}});
    legend.onAdd=()=>{{const div=L.DomUtil.create('div','legend');div.innerHTML='<strong>LOS velocity · mm/yr</strong><div class="gradient"></div><div style="display:flex;justify-content:space-between;font-size:10px;margin-top:3px"><span>subsidence</span><span>0</span><span>uplift</span></div>';return div;}};
    legend.addTo(map);

    Promise.all([
      addRaster(REPORT.rasters.primary,'velocity',REPORT.rasters.primaryLabel),
      addRaster(REPORT.rasters.compare,'coherence',REPORT.rasters.compareLabel)
    ]).then(([left,right])=>{{
      const bounds=left.getBounds(); map.fitBounds(bounds.pad(.06));
      L.control.sideBySide(left,right).addTo(map);
    }}).catch(error=>{{
      const notice=L.control({{position:'topright'}}); notice.onAdd=()=>{{const d=L.DomUtil.create('div','legend');d.textContent=`Could not read embedded COG: ${{error.message}}`;return d;}};notice.addTo(map);
    }});
  </script>
</body>
</html>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-name", required=True, help="Display name for the map report")
    parser.add_argument("--stack-dir", type=Path, required=True, help="Stack directory containing COGs and summary")
    parser.add_argument("--corridors", type=Path, required=True, help="Corridor GeoJSON for this site")
    parser.add_argument("--gnss-json", type=Path, help="Optional GNSS validation report JSON")
    parser.add_argument(
        "--gnss-cache-dir",
        type=Path,
        default=Path("insar_work/gnss/unr"),
        help="Optional directory containing station .NA.tenv3 histories",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output standalone HTML file")
    parser.add_argument("--primary-layer", default="velocity_mm_yr.tif", help="Primary COG filename (default: velocity_mm_yr.tif)")
    parser.add_argument("--compare-layer", default="temporal_coherence.tif", help="Swipe comparison COG filename (default: temporal_coherence.tif)")
    parser.add_argument("--primary-label", default="LOS velocity (mm/yr)")
    parser.add_argument("--compare-label", default="Temporal coherence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    gnss_cache_dir = args.gnss_cache_dir if args.gnss_cache_dir.is_dir() else None
    output = build_report(
        site_name=args.site_name,
        stack_dir=args.stack_dir,
        corridors_path=args.corridors,
        validation_path=args.gnss_json,
        gnss_cache_dir=gnss_cache_dir,
        output_path=args.output,
        primary_layer=args.primary_layer,
        compare_layer=args.compare_layer,
        primary_label=args.primary_label,
        compare_label=args.compare_label,
    )
    print(f"Wrote interactive report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
