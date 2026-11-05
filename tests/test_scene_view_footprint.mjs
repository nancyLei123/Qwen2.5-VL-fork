import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import * as THREE from '../vendor/gaussian-viewer/node_modules/three/build/three.module.js';
import {traceFootprint,validateFootprint,CameraFootprint} from '../scripts/templates/scene_view_footprint.mjs';

const origin=[0,0,0,0,0,0,1], rays=[[0,0,-1]], patch=z=>[0,0,z,0,0,1,.05];
const base=new URL('../runs/RoomScan3c/environment_reconstruction/',import.meta.url);
const haveArtifacts=['scene_playback_data.json','view_footprint/footprint_data.json']
  .every(name=>fs.existsSync(new URL(name,base)));
const text=haveArtifacts ? fs.readFileSync(new URL('scene_playback_data.json',base),'utf8') : '';
const data=haveArtifacts ? JSON.parse(text) : null;
const payload=haveArtifacts ? JSON.parse(fs.readFileSync(new URL('view_footprint/footprint_data.json',base),'utf8')) : null;
const hash=createHash('sha256').update(text).digest('hex');
const artifactTest=(name,run)=>test(name,{skip:!haveArtifacts && 'Private RoomScan3c artifacts absent'},run);

artifactTest('footprint fails closed for another trial, model, pose export or lens table',()=>{
  assert.equal(validateFootprint(payload,data,hash),payload);
  for(const invalid of [{graph_uid:'other'},{model_sha256:'other'},{playback_sha256:'stale'},
    {trial_id:'Fuchs'},{camera:{...payload.camera,fx:700}},
    {frame_lut:[]},{rays:[[0,0,1]]},{patches:[[0,0,-2,0,0,1,10]]}])
    assert.throws(()=>validateFootprint({...payload,...invalid},data,hash));
});

test('nearest finite disc wins; no far wall through a supported foreground disc',()=>{
  const hits=traceFootprint(origin,[patch(-2),patch(-1)],rays,3);
  assert.deepEqual(hits[0].position,[0,0,-1]);assert.equal(hits[0].distance,1);
});

test('holes, behind-camera geometry, grazing incidence and range overflow are unknown',()=>{
  for(const patches of [[],[patch(2)],[patch(-4)],[[.2,0,-2,0,0,1,.05]],[[0,0,-2,1,0,0,.05]]])
    assert.deepEqual(traceFootprint(origin,patches,rays,3),[null]);
  assert.deepEqual(traceFootprint(null,[patch(-2)],rays,3),[null]);
  assert.deepEqual(traceFootprint([0,0,0,0,0,0,0],[patch(-2)],rays,3),[null]);
});

test('rotation, translation, rewind and scaled quaternion preserve the measured origin',()=>{
  const q=new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),Math.PI/2).toArray();
  const p=[3,2,1,...q.map(v=>2*v)], before=[...p];
  const hits=traceFootprint(p,[[1,2,1,1,0,0,.05]],rays,3);
  assert.ok(Math.abs(hits[0].distance-2)<1e-9);
  assert.deepEqual(p,before);
  assert.equal(traceFootprint(origin,[patch(-2)],rays,3)[0].distance,2);
});

artifactTest('cone anchors to head; video mapping stays available in first person; gaps clear all candidates',()=>{
  const el=()=>({textContent:'',value:'',disabled:true,hidden:true,dataset:{},replaceChildren(){},toggleAttribute(name,value){if(name==='hidden')this.hidden=value;}});
  const elements=Object.fromEntries(['mode','reach','reachLabel','status','audit','overlay','stage'].map(k=>[k,el()]));
  const view=new CameraFootprint(THREE,data,new THREE.Scene(),elements);
  view.drawVideo=uv=>{assert.equal(uv.length,113);};view.install(payload);
  assert.equal(view.mode,'surface');
  view.mode='cone';
  const f=data.poses.findIndex(Boolean), p=data.poses[f];
  view.update(f,p,'room',true,true);
  assert.equal(view.root.visible,true);
  assert.equal(view.probes.count,0);
  assert.deepEqual(view.cone.position.toArray(),p.slice(0,3));
  view.update(f,p,'follow',true,true);
  assert.equal(view.root.visible,false);assert.equal(elements.overlay.hidden,false);
  view.update(f,p,'room',true,false);
  assert.equal(view.root.visible,false);assert.equal(elements.overlay.hidden,true);
  view.update(f,p,'room',true,true);view.mode='off';view.update(f,p,'room',true,true);
  assert.equal(view.root.visible,false);assert.equal(view.probes.count,0);
  view.mode='footprint';view.update(0,null,'room',true,true);
  assert.equal(view.root.visible,false);assert.equal(view.probes.count,0);
  assert.equal(elements.overlay.hidden,true);assert.equal(elements.stage.dataset.footprintPose,'');
});
