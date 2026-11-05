import {validFigurePose} from './scene_pose_figure.mjs';

export async function sha256Text(text) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(bytes)].map(v => v.toString(16).padStart(2, '0')).join('');
}

export function validateFootprint(payload, data, hash) {
  if (payload.schema_version !== 1 || payload.trial_id !== 'RoomScan3c' || payload.trial_id !== data.trial_id
      || payload.graph_uid !== data.provenance.graph_uid || payload.units !== 'meters' || payload.up_axis !== 'Z'
      || payload.playback_sha256 !== hash || payload.model_sha256 !== data.provenance.model_sha256
      || payload.fps !== data.fps || payload.frame_count !== data.poses.length)
    throw new Error('Footprint calibration/geometry does not match this exact recording and model.');
  if (JSON.stringify(payload.camera) !== JSON.stringify(data.scenes[0]?.camera)
      || payload.camera.width !== 1408 || payload.camera.height !== 1408
      || payload.camera.fx !== 704 || payload.camera.fy !== 704)
    throw new Error('Footprint crop does not match the reconstruction camera.');
  if (payload.audit?.status !== 'coordinate_checks_passed_surface_unvalidated'
      || payload.audit.video_lut_max_error_pixels > .21 || payload.audit.max_pose_matrix_difference > 1e-6)
    throw new Error('Coordinate alignment checks have not passed.');
  const finite = row => Array.isArray(row) && row.every(Number.isFinite);
  if (!payload.rays?.length || !payload.rays.every(r => finite(r) && r.length === 3 && r[2] < 0 && Math.abs(Math.hypot(...r)-1) < 1e-5)
      || payload.hit_ray_count !== 81 || payload.rays.length !== 113
      || !payload.patches?.length || !payload.patches.every(p => finite(p) && p.length === 7 && p[6] > 0 && p[6] <= .056
        && Math.abs(Math.hypot(...p.slice(3,6))-1) < 1e-4)) throw new Error('Invalid footprint rays or finite surface patches.');
  if (payload.frame_lut?.length !== data.poses.length || !payload.video_uv_lut?.length
      || !payload.video_uv_lut.every(row => row.length === payload.rays.length && row.every(p => finite(p) && p.length === 2 && p.every(v => v >= 0 && v <= 1)))
      || !payload.frame_lut.every(i => Number.isInteger(i) && i >= -1 && i < payload.video_uv_lut.length))
    throw new Error('Invalid frame-specific fisheye projection lookup.');
  return payload;
}

function rotate(v, q) {
  const [x,y,z,w] = q, [a,b,c] = v;
  const tx = 2*(y*c-z*b), ty = 2*(z*a-x*c), tz = 2*(x*b-y*a);
  return [a+w*tx+y*tz-z*ty, b+w*ty+z*tx-x*tz, c+w*tz+x*ty-y*tx];
}

export function traceFootprint(pose, patches, rays, reach) {
  if (!validFigurePose(pose) || !Number.isFinite(reach) || reach <= .2) return rays.map(() => null);
  const length = Math.hypot(...pose.slice(3)), q = pose.slice(3).map(v => v/length);
  const inverse = [-q[0],-q[1],-q[2],q[3]], origin = pose.slice(0,3);
  // Transform once per patch, then cast the small shared ray grid locally.
  const candidates = [];
  for (const p of patches) {
    const c = rotate([p[0]-origin[0],p[1]-origin[1],p[2]-origin[2]],inverse), radius=p[6];
    const depth = -c[2];
    if (depth+radius < .2 || depth-radius > reach || Math.abs(c[0]) > depth+2*radius || Math.abs(c[1]) > depth+2*radius) continue;
    const n = rotate(p.slice(3,6),inverse);
    candidates.push([...c,...n,radius, c[0]*n[0]+c[1]*n[1]+c[2]*n[2]]);
  }
  return rays.map((r,index) => {
    let nearest = Infinity, selected = null;
    for (const p of candidates) {
      const dot = r[0]*p[3]+r[1]*p[4]+r[2]*p[5];
      if (Math.abs(dot) < .25) continue;
      const t = p[7]/dot, depth=-r[2]*t;
      if (t >= nearest || depth < .2 || depth > reach) continue;
      const dx=r[0]*t-p[0],dy=r[1]*t-p[1],dz=r[2]*t-p[2];
      if (dx*dx+dy*dy+dz*dz > p[6]*p[6]) continue;
      nearest=t; selected=p;
    }
    if (!selected) return null;
    const offset=rotate(r.map(v=>v*nearest),q);
    return {index,position:offset.map((v,i)=>v+origin[i]),normal:rotate(selected.slice(3,6),q),distance:nearest,radius:selected[6]};
  });
}

export class CameraFootprint {
  constructor(THREE, data, scene, elements) {
    this.T=THREE; this.data=data; this.el=elements; this.mode='surface'; this.reach=3;
    this.root=new THREE.Group(); this.cone=new THREE.Group(); this.root.add(this.cone); scene.add(this.root);
    this.root.visible=false; this.key=''; this.hits=[];
    this.dummy=new THREE.Object3D(); this.axis=new THREE.Vector3(0,0,1);
    const material=new THREE.MeshBasicMaterial({color:'#f2cb75',transparent:true,opacity:.045,side:THREE.DoubleSide,depthWrite:false,depthTest:false});
    this.volume=new THREE.Mesh(new THREE.BufferGeometry(),material); this.cone.add(this.volume);
    this.lines=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#f2cb75',transparent:true,opacity:.65,depthTest:false}));
    this.cone.add(this.lines);
    this.probes=new THREE.InstancedMesh(new THREE.CircleGeometry(1,16),
      new THREE.MeshBasicMaterial({color:'#ffffff',side:THREE.DoubleSide,transparent:true,opacity:.8,depthTest:false,depthWrite:false}),81);
    this.probes.frustumCulled=false; this.probes.count=0; this.probes.renderOrder=22; this.root.add(this.probes);
    this.centerRay=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineDashedMaterial({color:'#f58f78',dashSize:.035,gapSize:.025,depthTest:false}));
    this.root.add(this.centerRay);
    this.el.mode.onchange=()=>{this.mode=this.el.mode.value;this.key='';};
    this.el.reach.oninput=()=>{this.reach=Number(this.el.reach.value);this.key='';};
  }

  install(payload) {
    this.payload=payload;
    this.rays=payload.rays.slice(0,payload.hit_ray_count);
    this.corners=[payload.rays[81],payload.rays[89],payload.rays[97],payload.rays[105]];
    this.probes.instanceMatrix.setUsage(this.T.DynamicDrawUsage);
    this.el.audit.textContent=`Internal checks passed: ${payload.audit.pose_checks} camera-pose comparisons; all ${payload.audit.supported_frame_projections_checked.toLocaleString()} supported frames lens-checked; lookup error ≤ ${payload.audit.video_lut_max_error_pixels.toFixed(2)} source pixels. ${payload.patches.length.toLocaleString()} locally fitted patches. This is coordinate consistency, NOT independent surface or gaze validation.`;
    this.el.mode.disabled=this.el.reach.disabled=false;
    this.el.mode.value=this.mode;
  }

  update(frame,pose,viewMode,ready,videoReady=true) {
    const p=this.payload;
    this.el.reach.disabled=this.mode==='surface' || this.mode==='off' || !p;
    this.el.reachLabel.textContent=this.mode==='surface' ? 'Surface depth: model estimate' : `${this.reach.toFixed(1)} m / display depth`;
    const supported=p && p.frame_lut[frame]>=0 && validFigurePose(pose);
    const visible=supported && ready && this.mode!=='off';
    this.root.visible=Boolean(visible && viewMode!=='follow' && this.mode!=='surface');
    this.el.overlay.toggleAttribute('hidden', !visible || !videoReady);
    if (!p) return;
    if (!visible || !videoReady) {
      this.root.visible=false;
      this.hits=[]; this.probes.count=0;this.centerRay.visible=false;this.key='';this.el.overlay.replaceChildren();
      this.el.stage.dataset.footprintHits='0';this.el.stage.dataset.footprintCenter='';
      this.el.stage.dataset.footprintPose='';this.el.stage.dataset.footprintFrame='';
      this.el.status.textContent=this.mode==='off' ? 'Camera-to-environment overlay is off.'
        : !supported ? 'No supported camera pose/calibration: no cone or intersections inferred.'
          : !videoReady ? 'Waiting for the video frame; candidate footprint is hidden until synchronized.'
            : 'Waiting for the matching reconstructed scene.';
      return;
    }
    this.root.visible=this.root.visible && videoReady;
    this.cone.position.fromArray(pose); this.cone.quaternion.fromArray(pose,3).normalize();
    const key=`${frame}:${this.mode}:${this.reach}`;
    if (key===this.key) return;
    this.key=key;
    const corners=this.corners.map(r=>r.map(v=>v*this.reach/-r[2]));
    const faces=[],lines=[];
    for(let i=0;i<4;i++){ faces.push(0,0,0,...corners[i],...corners[(i+1)%4]); lines.push(0,0,0,...corners[i]); }
    for(const [object,vertices] of [[this.volume,faces],[this.lines,lines]]) {
      object.geometry.dispose(); object.geometry=new this.T.BufferGeometry();
      object.geometry.setAttribute('position',new this.T.Float32BufferAttribute(vertices,3));
    }
    this.hits=this.mode==='footprint' ? traceFootprint(pose,p.patches,this.rays,this.reach) : this.rays.map(()=>null);
    let count=0;
    for(const hit of this.hits) {
      if(!hit) continue;
      this.dummy.position.fromArray(hit.position);
      this.dummy.quaternion.setFromUnitVectors(this.axis,new this.T.Vector3(...hit.normal).normalize());
      this.dummy.scale.setScalar(Math.min(.04,hit.radius));this.dummy.updateMatrix();
      this.probes.setMatrixAt(count,this.dummy.matrix);
      this.probes.setColorAt(count,new this.T.Color(hit.index===40?'#f58f78':'#56dbd0'));count++;
    }
    this.probes.count=count;this.probes.instanceMatrix.needsUpdate=true;
    if(this.probes.instanceColor)this.probes.instanceColor.needsUpdate=true;
    const center=this.hits[40];this.centerRay.visible=Boolean(center);
    if(center){this.centerRay.geometry.dispose();this.centerRay.geometry=new this.T.BufferGeometry().setFromPoints([new this.T.Vector3(...pose.slice(0,3)),new this.T.Vector3(...center.position)]);this.centerRay.computeLineDistances();}
    this.drawVideo(p.video_uv_lut[p.frame_lut[frame]]);
    if(this.mode!=='surface')this.el.status.textContent=this.mode==='cone' ? 'Calibrated central 90° crop; cone length is a display limit, not a surface distance. Not eye gaze.'
      : `${count}/${this.rays.length} rays meet local landmark patches. ${center?`Center candidate: ${center.distance.toFixed(2)} m.`:'Center: no supported patch, not necessarily empty space.'} Approximate geometry; occlusion is NOT verified.`;
    this.el.stage.dataset.footprintFrame=frame;
    this.el.stage.dataset.footprintHits=count;
    this.el.stage.dataset.footprintCenter=center?JSON.stringify(center):'';
    this.el.stage.dataset.footprintPose=JSON.stringify(pose);
  }

  drawVideo(uv) {
    const ns='http://www.w3.org/2000/svg', nodes=[];
    const boundary=document.createElementNS(ns,'polygon');
    boundary.setAttribute('points',uv.slice(this.payload.hit_ray_count).map(([u,v])=>`${u*1000},${v*1000}`).join(' '));
    boundary.setAttribute('class','crop-boundary');nodes.push(boundary);
    if(this.mode==='footprint')for(let i=0;i<this.rays.length;i++){
      const dot=document.createElementNS(ns,'circle');dot.setAttribute('cx',uv[i][0]*1000);dot.setAttribute('cy',uv[i][1]*1000);
      dot.setAttribute('r',this.hits[i]?5:2);dot.setAttribute('class',this.hits[i]?(i===40?'center-probe':'hit-probe'):'unknown-probe');
      const title=document.createElementNS(ns,'title');title.textContent=this.hits[i]?`Candidate ${i+1}: ${this.hits[i].distance.toFixed(2)} m, unvalidated surface`:`Probe ${i+1}: no supported patch`;
      dot.append(title);nodes.push(dot);
    }
    this.el.overlay.replaceChildren(...nodes);
  }
}
