import {routeSegments} from './scene_route_navigation.mjs';

export function validateSpatialMap(map, data) {
  if (map.trial_id !== data.trial_id || map.graph_uid !== data.provenance.graph_uid)
    throw new Error('Landmarks and playback do not share the same trial and MPS world frame.');
  if (map.units !== 'meters' || map.up_axis !== 'Z' || !Array.isArray(map.points) || !map.points.length)
    throw new Error('No usable metric MPS landmarks.');
  if (!map.points.every(p => p.length === 3 && p.every(Number.isFinite)))
    throw new Error('Invalid landmark coordinates.');
  return map.points;
}

export function spatialBounds(points, padding = .04) {
  const bounds = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]];
  for (const p of points) for (let j = 0; j < 3; j++) {
    bounds[0][j] = Math.min(bounds[0][j], p[j]); bounds[1][j] = Math.max(bounds[1][j], p[j]);
  }
  return [0, 1, 2].map(j => {
    const pad = Math.max(.25, (bounds[1][j] - bounds[0][j]) * padding);
    return [bounds[0][j] - pad, bounds[1][j] + pad];
  });
}

export function cameraWire(pose, depth = .8) {
  if (!pose) return {corners: [], edges: [], forward: []};
  // Exported camera quaternions use the viewer convention: -Z forward, +Y up.
  const q = pose.slice(3), norm = Math.hypot(...q);
  if (!norm || !pose.every(Number.isFinite)) return {corners: [], edges: [], forward: []};
  const [x, y, z, w] = q.map(v => v / norm), origin = pose.slice(0, 3);
  const transform = ([vx, vy, vz]) => {
    const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
    return [origin[0] + vx + w * tx + y * tz - z * ty,
      origin[1] + vy + w * ty + z * tx - x * tz,
      origin[2] + vz + w * tz + x * ty - y * tx];
  };
  const corners = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b]) => transform([a*depth, b*depth, -depth]));
  const edges = corners.flatMap((p,i) => [origin,p,null,p,corners[(i+1)%4],null]);
  return {corners, edges, forward: [origin, transform([0,0,-depth*1.6])]};
}

export function routeArrays(edges) {
  const points = [], times = [];
  for (const edge of edges) { points.push(edge.a.position, edge.b.position, null); times.push(edge.a.time, edge.b.time, null); }
  return {...xyz(points), customdata: times};
}

export function routePickTime(point) {
  if (!point || point.curveNumber < 1 || point.curveNumber > 3) return null;
  // Plotly's 3D click payload can omit customdata even when its hover has it.
  const value = point.customdata ?? point.data?.customdata?.[point.pointNumber];
  return Number.isFinite(value) ? value : null;
}

function xyz(points) {
  return {x: points.map(p => p?.[0] ?? null), y: points.map(p => p?.[1] ?? null), z: points.map(p => p?.[2] ?? null)};
}
const defaultCamera = () => ({eye: {x:1.35,y:-1.55,z:1.05}, up:{x:0,y:0,z:1}, center:{x:0,y:0,z:0}});

async function loadPlotly() {
  if (globalThis.Plotly) return globalThis.Plotly;
  await new Promise((resolve,reject) => {
    const script = document.createElement('script'); script.src = 'spatial_plotly.min.js';
    script.onload = resolve; script.onerror = () => reject(new Error('The local 3D plotting library could not load.'));
    document.head.appendChild(script);
  });
  return globalThis.Plotly;
}

export class SpatialOverview {
  constructor(data, seek) {
    this.data = data; this.seek = seek; this.time = 0; this.pose = null;
    this.range = [0,data.duration_s]; this.rangeKey = ''; this.dirtyRoute = true;
    this.visible = true; this.mode = '3d'; this.scope = 'all'; this.camera = defaultCamera();
    this.$ = id => document.getElementById(id);
    this.plot = this.$('spatialPlot'); this.note = this.$('spatialStatus');
    this.$('spatialMode').onchange = () => this.setMode(this.$('spatialMode').value);
    this.$('spatialWhole').onclick = () => this.fit(false);
    this.$('spatialFocus').onclick = () => this.fit(true);
    this.$('spatialTop').onclick = () => this.setCamera({eye:{x:0,y:0,z:2.1},up:{x:0,y:1,z:0},center:{x:0,y:0,z:0}});
    this.$('spatialIn').onclick = () => this.zoom(.8);
    this.$('spatialOut').onclick = () => this.zoom(1.25);
    this.$('spatialPoints').onchange = () => { if(this.ready) this.P.restyle(this.plot,{visible:this.$('spatialPoints').checked},[0]); };
    this.observer = new IntersectionObserver(([entry]) => { this.visible = entry.isIntersecting; this.schedule(); });
    this.observer.observe(this.plot);
  }

  async init() {
    try {
      const response = await fetch('spatial_overview_data.json',{cache:'no-cache'});
      if (!response.ok) throw new Error(`Landmark map not exported (HTTP ${response.status}).`);
      const map = await response.json(); this.points = validateSpatialMap(map,this.data);
      this.P = await loadPlotly();
      const path = routeSegments(this.data,0,this.data.duration_s,.5);
      this.globalRanges = spatialBounds([...this.points,...path.flatMap(e=>[e.a.position,e.b.position])]);
      const line = (name,color,width=5) => ({type:'scatter3d',mode:'lines',name,...xyz([]),line:{color,width},hoverinfo:'skip',connectgaps:false});
      const routes = {...line('Recorded route','#556b64',4),...routeArrays(path),
        hovertemplate:'Video %{customdata:.1f} s<extra>Click to seek</extra>'};
      const timedLine = (name,color) => ({...line(name,color,7),hoverinfo:undefined,
        hovertemplate:'Video %{customdata:.1f} s<extra>Click to seek</extra>'});
      const traces = [
        {type:'scatter3d',mode:'markers',name:'Measured MPS landmarks',...xyz(this.points),
          marker:{size:1.8,opacity:.6,color:this.points.map(p=>p[2]),colorscale:'Viridis',
            colorbar:{title:'Z (m)',thickness:10,len:.6,x:1.01},cmin:this.globalRanges[2][0],cmax:this.globalRanges[2][1]},hoverinfo:'skip'},
        routes, timedLine('Recent path','#128868'), timedLine('Upcoming path','#d29029'),
        {type:'scatter3d',mode:'markers',name:'Recorded camera',...xyz([]),marker:{size:7,color:'#06a57a',line:{color:'#fff',width:2}},hoverinfo:'skip'},
        line('Recorded viewing rectangle','#d28512',5), line('Facing direction','#d46e21',6),
      ];
      const axis = (name,range) => ({title:name,range,autorange:false,backgroundcolor:'#f5f7f2',showbackground:true,
        gridcolor:'#dce5dc',zerolinecolor:'#b1c1b3',ticks:'outside',tickfont:{size:11},showspikes:false});
      await this.P.newPlot(this.plot,traces,{
        paper_bgcolor:'#f8faf5',margin:{l:30,r:50,t:5,b:15},showlegend:false,
        font:{family:'Avenir Next, sans-serif',color:'#344c41'},uirevision:'spatial-overview',
        scene:{xaxis:axis('World X (m)',this.globalRanges[0]),yaxis:axis('World Y (m)',this.globalRanges[1]),
          zaxis:axis('World Z / up (m)',this.globalRanges[2]),aspectmode:'data',camera:this.camera,dragmode:'orbit'},
      },{displayModeBar:false,scrollZoom:false,responsive:true});
      this.ready = true; this.plot.dataset.ready = 'true'; this.plot.dataset.landmarks = this.points.length;
      this.plot.dataset.camera = JSON.stringify(this.camera);
      this.plot.dataset.ranges = JSON.stringify(this.globalRanges);
      this.note.textContent = `${this.points.length.toLocaleString()} measured landmarks. Drag to rotate; use + / - to zoom. Click the route to seek. The overview stays still during playback.`;
      this.plot.on('plotly_relayout', event => {
        if(event['scene.camera']) { this.camera = event['scene.camera']; this.plot.dataset.camera = JSON.stringify(this.camera); }
      });
      this.plot.on('plotly_click', event => {
        const time = event.points?.map(routePickTime).find(value=>value!==null);
        if(time!==undefined) this.seek(time);
      });
      this.resize = new ResizeObserver(() => { if(this.mode==='3d') this.P.Plots.resize(this.plot); });
      this.resize.observe(this.plot);
      for(const el of document.querySelectorAll('[data-spatial-control]')) el.disabled = false;
      this.schedule();
    } catch(error) {
      this.note.textContent = `3D overview unavailable: ${error.message} The 2D route and synchronized player remain available.`;
      this.setMode('2d');
    }
  }

  setMode(mode) {
    this.mode = mode; this.$('spatialMode').value = mode;
    this.plot.hidden = mode !== '3d'; this.$('routeMap').hidden = mode !== '2d';
    this.$('spatialTools').hidden = mode !== '3d';
    if(this.ready && mode==='3d') { this.P.Plots.resize(this.plot); this.schedule(); }
  }

  setRange(range) {
    const key = range.join(':'); if(key===this.rangeKey) return;
    this.range = range; this.rangeKey = key; this.dirtyRoute = true; this.schedule();
  }

  update(time,pose) {
    if(time===this.time && pose===this.pose) return;
    this.time = time; this.pose = pose; this.schedule();
  }

  schedule() {
    if(!this.ready || !this.visible || this.mode!=='3d' || this.timer || this.busy) return;
    this.timer = setTimeout(()=>{this.timer=null;this.render();},Math.max(0,120-(performance.now()-(this.lastRender||0))));
  }

  async render() {
    this.busy = true;
    const time = this.time, pose = this.pose, rangeKey = this.rangeKey;
    try {
      if(this.dirtyRoute) {
        this.dirtyRoute = false;
        const route = routeArrays(routeSegments(this.data,...this.range,.5));
        await this.P.restyle(this.plot,Object.fromEntries(Object.entries(route).map(([key,val])=>[key,[val]])),[1]);
      }
      const recent = routeArrays(routeSegments(this.data,Math.max(this.range[0],time-15),Math.min(time,this.range[1]),.2));
      const next = routeArrays(routeSegments(this.data,Math.max(time,this.range[0]),Math.min(this.range[1],time+15),.2));
      const wire = cameraWire(pose);
      const items = [recent,next,xyz(pose?[pose]:[]),xyz(wire.edges),xyz(wire.forward)];
      await this.P.restyle(this.plot,{x:items.map(p=>p.x),y:items.map(p=>p.y),z:items.map(p=>p.z),
        customdata:[recent.customdata,next.customdata,[],[],[]]},[2,3,4,5,6]);
      this.plot.dataset.time = String(time); this.plot.dataset.pose = pose ? JSON.stringify(pose) : '';
      this.$('spatialPoseNote').textContent = !pose ? 'No supported camera pose at this time; no position or direction is drawn.'
        : this.scope==='local' && pose.slice(0,3).some((v,j)=>v<this.ranges[j][0]||v>this.ranges[j][1])
          ? 'The wearer has moved outside this fixed view. Click Focus current position to recenter.'
          : 'Green dot + gold rectangle: recorded camera position and facing direction (schematic 90-degree view).';
    } catch(error) {
      this.note.textContent = `Could not refresh the 3D overview: ${error.message}`;
    } finally {
      this.busy = false; this.lastRender = performance.now();
      if(time!==this.time || rangeKey!==this.rangeKey) this.schedule();
    }
  }

  fit(local) {
    if(!this.ready) return;
    if(local && !this.pose) { this.$('spatialPoseNote').textContent = 'No supported pose here. Choose a time with camera data to focus.'; return; }
    this.scope = local ? 'local' : 'all';
    this.ranges = local ? this.pose.slice(0,3).map((value,j)=>[value-(j===2?4:8),value+(j===2?4:8)]) : this.globalRanges;
    const changes = {'scene.camera':defaultCamera()};
    ['x','y','z'].forEach((axis,j)=>{changes[`scene.${axis}axis.range`]=this.ranges[j];});
    this.P.relayout(this.plot,changes); this.camera=defaultCamera();
    this.$('spatialWhole').classList.toggle('active',!local); this.$('spatialFocus').classList.toggle('active',local);
    this.plot.dataset.scope = this.scope; this.schedule();
    this.plot.dataset.ranges = JSON.stringify(this.ranges);
  }

  setCamera(camera) { if(this.ready) { this.camera = camera; this.P.relayout(this.plot,{'scene.camera':camera}); } }
  zoom(factor) {
    const camera = structuredClone(this.camera), eye = camera.eye;
    const distance = Math.hypot(eye.x,eye.y,eye.z), target = Math.min(6,Math.max(.15,distance*factor));
    for(const key of ['x','y','z']) eye[key] *= target/distance;
    this.setCamera(camera);
  }
}
