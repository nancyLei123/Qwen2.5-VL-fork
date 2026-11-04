import * as THREE from 'three';
const $ = id => document.getElementById(id);
const fmt = s => `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`;
let data, viewer, camera, overlay, eye, frustum, marker, mode='follow', loading=false;
let currentModel=null;
const fetchJSON = async name => {const r=await fetch(name,{cache:'no-store'}); if(!r.ok) throw Error(`${name}: ${r.status}`); return r.json();};
function pose(frame){const m=new THREE.Matrix4().set(...frame.T_world_camera.flat());return m.multiply(new THREE.Matrix4().makeRotationX(Math.PI));}
function choose(index){
  if(!data) return;
  const i=Math.max(0,Math.min(data.frames.length-1,Number(index))); $('timeline').value=i;
  const f=data.frames[i]; $('reference').src=`images/${f.name}`; $('frameTime').textContent=fmt(f.recording_s);
  $('time').textContent=`${fmt(f.recording_s)} / ${fmt(data.duration_s)} | view ${i+1} of ${data.frames.length}`;
  if(!viewer) return;
  const m=pose(f), p=new THREE.Vector3(), q=new THREE.Quaternion(), s=new THREE.Vector3();m.decompose(p,q,s);
  eye.position.copy(p);eye.quaternion.copy(q);eye.updateMatrixWorld(true);frustum.update();marker.position.copy(p);
  const forward=new THREE.Vector3(0,0,-1).applyQuaternion(q), up=new THREE.Vector3(0,1,0).applyQuaternion(q);
  if(mode==='follow'){
    viewer.controls.enabled=false;camera.position.copy(p);camera.quaternion.copy(q);camera.up.copy(up);
    frustum.visible=false;marker.visible=false;
  }else{
    viewer.controls.enabled=true;camera.up.set(0,0,1);camera.position.copy(p).addScaledVector(forward,-.6).add(new THREE.Vector3(0,0,.12));
    viewer.controls.target.copy(p);camera.lookAt(p);viewer.controls.update();frustum.visible=true;marker.visible=true;
  }
  camera.updateMatrixWorld(true);
}
async function loadModel(url){
  if(!url||loading||currentModel===url)return;
  loading=true;$('notice').textContent='Loading local Gaussian model...';
  try{
    if(!viewer){
      const G=await import('/Qwen2.5-VL-fork/vendor/gaussian-viewer/node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js');
      camera=new THREE.PerspectiveCamera(90,1,.04,80);camera.up.set(0,0,1);
      overlay=new THREE.Scene();
      viewer=new G.Viewer({rootElement:$('scene'),camera,selfDrivenMode:false,sharedMemoryForWorkers:false,
        gpuAcceleratedSort:false,integerBasedSort:false,ignoreDevicePixelRatio:false,sphericalHarmonicsDegree:2,
        sceneRevealMode:G.SceneRevealMode.Instant,renderMode:G.RenderMode.OnChange,logLevel:G.LogLevel.None});
      viewer.controls.enableDamping=false;viewer.controls.minDistance=.15;viewer.controls.maxDistance=6;
      viewer.controls.enablePan=false;viewer.onMouseClick=()=>{};
      // The renderer updates OrbitControls even when disabled; preserve the recorded pose.
      const updateOrbit=viewer.controls.update.bind(viewer.controls);
      viewer.controls.update=()=>mode==='around'?updateOrbit():false;
      eye=new THREE.PerspectiveCamera(90,1,.04,.45);frustum=new THREE.CameraHelper(eye);
      const gold=new THREE.Color('#dfb953');frustum.setColors(gold,gold,gold,gold,gold);frustum.material.depthTest=false;overlay.add(frustum);
      marker=new THREE.Mesh(new THREE.SphereGeometry(.035,12,8),new THREE.MeshBasicMaterial({color:'#2bd6a0',depthTest:false}));overlay.add(marker);
      const animate=()=>{requestAnimationFrame(animate);if(!viewer)return;
        viewer.update();
        if(viewer.shouldRender()){
          viewer.render();viewer.renderer.autoClear=false;viewer.renderer.clearDepth();
          viewer.renderer.render(overlay,camera);viewer.renderer.autoClear=true;
        }
        viewer.renderNextFrame=false;
      };animate();
    }
    if(currentModel)await viewer.removeSplatScene(0);
    await viewer.addSplatScene(url,{splatAlphaRemovalThreshold:5,showLoadingUI:true,progressiveLoad:false});
    currentModel=url;choose($('timeline').value);
    $('checkpoint').value=url;
    const label=data.models.find(m=>m.url===url)?.label??'Joint room';
    $('notice').textContent=`${label} | measured MPS coordinates | compare against the reference image`;
  }catch(e){$('notice').textContent=`Viewer error: ${e.message}`;console.error(e);}
  finally{loading=false;}
}
function comparison(){
  if(!data.evaluations?.length)return;
  const es=data.evaluations;
  const table=document.createElement('div');table.className='overflow';
  table.innerHTML='<table><thead><tr><th>Checkpoint</th><th>Background PSNR</th><th>Background SSIM</th><th>Full-frame PSNR</th></tr></thead><tbody></tbody></table>';
  for(const e of es){const tr=document.createElement('tr');for(const x of [e.step.toLocaleString(),e.mean_background_psnr_db.toFixed(2)+' dB',e.mean_background_ssim.toFixed(3),e.mean_psnr_db.toFixed(2)+' dB']){const td=document.createElement('td');td.textContent=x;tr.append(td);}table.querySelector('tbody').append(tr);}
  $('comparison').replaceChildren(table);
  if(data.quality_summary){
    const q=data.quality_summary, p=document.createElement('p');
    p.textContent=`30,000 vs 10,000: ${q.psnr_improved} of ${q.views} views improved in background PSNR (${q.mean_background_psnr_gain_db.toFixed(2)} dB mean gain); ${q.ssim_improved} improved in background SSIM. Gains are modest and not consistent at every viewpoint. Small text, thin edges and reflective objects can remain soft.`;
    $('comparison').append(p);
  }
  for(let i=0;i<es[0].views.length;i++){
    const row=document.createElement('div');row.className='comparison-row';
    const h=document.createElement('b');h.textContent=`Held-out view at ${fmt(es[0].views[i].recording_s)}`;row.append(h);
    const grid=document.createElement('div');grid.className='compare';
    const figures=[{src:`images/${es[0].views[i].name}`,title:'Actual Aria reference'},...es.map(e=>({src:e.views[i].render_relative,title:`${e.step.toLocaleString()} iterations`}))];
    for(const f of figures){const fig=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src=f.src;im.loading='lazy';im.alt=f.title;cap.textContent=f.title;fig.append(im,cap);grid.append(fig);}row.append(grid);$('comparison').append(row);
  }
}
async function refresh(){
  try{
    const p=await fetchJSON('progress.json');
    const n=p.step??p.done??p.candidates_done, total=p.total_steps??p.total??p.candidates_total;
    const pct=total?100*n/total:null;
    $('progressText').textContent=p.stage.replaceAll('_',' ')+(pct!==null?` | ${pct.toFixed(1)}% of this stage`:'');
    $('progressFill').style.width=p.state==='complete'?'100%':pct!==null?`${pct}%`:'0%';
    $('badge').textContent=p.state==='complete'?'Training + evaluation complete':'Local processing';
    const next=await fetchJSON('viewer_data.json');
    const changed=JSON.stringify(data?.models)!==JSON.stringify(next.models)||JSON.stringify(data?.evaluations)!==JSON.stringify(next.evaluations);
    data=next;
    if(changed){
      const selected=currentModel;$('checkpoint').replaceChildren();
      for(const m of data.models){const o=document.createElement('option');o.value=m.url;o.textContent=m.label;$('checkpoint').append(o);}
      $('checkpoint').disabled=!data.models.length;
      if(selected&&data.models.some(m=>m.url===selected))$('checkpoint').value=selected;
      comparison();
    }
    if(!currentModel&&data.models.length)await loadModel(data.models.at(-1).url);
  }catch(e){console.warn(e);$('progressText').textContent=`Status unavailable: ${e.message}`;}
}
try{
 data=await fetchJSON('viewer_data.json');$('timeline').max=data.frames.length-1;$('endTime').textContent=fmt(data.duration_s);
 for(const [value,title] of [[data.frames.length,'calibrated views'],[data.training_count,'training views'],[data.validation_count,'held-out views'],['1408 px','rectified image width']]){const d=document.createElement('div'),v=document.createElement('strong'),t=document.createElement('span');v.textContent=value;t.textContent=title;d.append(v,t);$('facts').append(d);}
 $('timeline').oninput=()=>choose($('timeline').value);$('prev').onclick=()=>choose(Number($('timeline').value)-1);$('next').onclick=()=>choose(Number($('timeline').value)+1);
 for(const [id,m] of [['follow','follow'],['around','around']])$(id).onclick=()=>{mode=m;$('follow').classList.toggle('active',m==='follow');$('around').classList.toggle('active',m==='around');choose($('timeline').value);};
 $('reset').onclick=()=>choose($('timeline').value);$('checkpoint').onchange=()=>loadModel($('checkpoint').value);
 choose(Math.floor(data.frames.length*.24));data.models=null;await refresh();setInterval(refresh,10000);
}catch(e){$('progressText').textContent=e.message;console.error(e);}
