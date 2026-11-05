import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from '../vendor/gaussian-viewer/node_modules/three/build/three.module.js';
import {acceptsSurfaceDepth,visibilityShader,GaussianSurfaceVisibility} from '../scripts/templates/scene_surface_visibility.mjs';

test('surface depth accepts supported foreground, rejects hidden background and unknowns',()=>{
  assert.equal(acceptsSurfaceDepth(2,2,2.01,[2.01,2,1.99,2]),true);
  for(const input of [[4,2,2],[2,-1,-1],[2,2,3],[2,2,2,[2,4]],
    [2,2,2,[-1]],[-2,2,2],[13,13,13],[NaN,2,2],[2,Infinity,2]])
    assert.equal(acceptsSurfaceDepth(...input),false);
});

test('shader changes color only and rejects an unknown renderer template',()=>{
  const shader='void main () { gl_FragColor = vec4(color.rgb, opacity); }';
  const result=visibilityShader(shader);
  assert.ok(result.includes('pdObserverInverseVP'));
  assert.ok(result.includes('pdRecordedVP*world'));
  assert.ok(result.includes('pdDepthHigh'));
  assert.equal(result.split('gl_FragColor = vec4(color.rgb, opacity);').length,2);
  assert.ok(result.includes('color=mix(color'));
  assert.throws(()=>visibilityShader('void main() {}'));
  assert.throws(()=>visibilityShader('gl_FragColor = vec4(color.rgb, opacity);'));
  assert.throws(()=>visibilityShader(shader+shader));
});

const identity={trial_id:'RoomScan3c',provenance:{model_sha256:'model',graph_uid:'graph'}};
const alignment={trial_id:'RoomScan3c',model_sha256:'model',graph_uid:'graph',frame_lut:[0,-1]};
const pose=[0,0,0,0,0,0,1];
const harness=()=>{
  const renderer={debug:{},getDrawingBufferSize:v=>v.set(100,100)};
  const viewer={renderer,camera:new THREE.PerspectiveCamera()};
  const el={stage:{dataset:{}},status:{textContent:''}};
  return {view:new GaussianSurfaceVisibility(THREE,viewer,identity,el),el};
};

test('another trial, graph, or model cannot activate a surface estimate',()=>{
  const {view}=harness();
  for(const patch of [{trial_id:'Fuchs'},{graph_uid:'other'},{model_sha256:'stale'}])
    assert.throws(()=>view.install({...alignment,...patch}));
  view.install(alignment);
});

test('seeking, missing poses, unsupported frames and Off clear stale highlights',()=>{
  const {view,el}=harness();view.install(alignment);
  let draws=0;
  view.attach=()=>{view.uniforms={pdSurfaceEnabled:{value:0}};};
  view.renderDepth=frame=>{view.key=String(frame);draws++;};
  view.update(0,pose,'surface',true,true);
  assert.equal(el.stage.dataset.surfaceVisible,'true');assert.equal(draws,1);
  view.update(0,pose,'surface',true,true);assert.equal(draws,1);
  for(const args of [[0,pose,'off',true,true],[0,pose,'surface',true,false],
    [1,pose,'surface',true,true],[0,null,'surface',true,true],[2,pose,'surface',true,true],
    [0,pose,'surface',false,true]]) {
    view.update(...args);
    assert.equal(el.stage.dataset.surfaceVisible,'false');
    assert.equal(view.uniforms.pdSurfaceEnabled.value,0);
    assert.equal(el.stage.dataset.surfacePose,'');
  }
});

test('depth render must include splats outside observer culling and restore GPU state',()=>{
  const text=fs.readFileSync(new URL('../scripts/templates/scene_surface_visibility.mjs',import.meta.url),'utf8');
  assert.ok(text.includes('Uint32Array.from({length:count},(_,i)=>i)'));
  assert.ok(text.includes('geometry.setDrawRange(0,6)'));
  for(const restore of ['setRenderTarget(previous.target)','setViewport(previous.viewport)',
    'setScissor(previous.scissor)','setScissorTest(previous.scissorTest)',
    'setClearColor(previous.color,previous.alpha)','autoClear=previous.autoClear'])assert.ok(text.includes(restore));
  assert.ok(text.includes('this.source.material=this.original'));
});

test('optional GPU failure restores the original material and remains explicit after toggling',()=>{
  const {view,el}=harness(), original={name:'original'};
  view.source={material:{name:'tint'}};view.original=original;
  const warn=console.warn;console.warn=()=>{};
  try{view.fail(new Error('test compilation error'));}finally{console.warn=warn;}
  assert.equal(view.source.material,original);
  el.status.textContent='Off';view.update(0,pose,'surface',true,true);
  assert.match(el.status.textContent,/test compilation error/);
  assert.equal(el.stage.dataset.surfaceVisible,'false');
});
