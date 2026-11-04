import test from 'node:test';
import assert from 'node:assert/strict';
import {waitForSplatTree, SceneQueue} from '../scripts/templates/scene_playback_core.mjs';

test('another scene cannot start until the renderer finishes its spatial tree', async () => {
  const mesh={baseSplatTree:{},tree:null,getSplatTree(){return this.tree;}};
  const loaded=[];
  const queue=new SceneQueue(async scene=>{
    loaded.push(scene.id);
    await waitForSplatTree(mesh,1000,1);
    return loaded.length-1;
  },()=>{},()=>{});
  queue.request({id:'first'});queue.request({id:'second'});
  await new Promise(resolve=>setTimeout(resolve,5));
  assert.deepEqual(loaded,['first']);
  mesh.baseSplatTree=null;mesh.tree={};
  await new Promise(resolve=>setTimeout(resolve,10));
  assert.deepEqual(loaded,['first','second']);
});

test('a stalled or disposed spatial index fails explicitly instead of hanging', async () => {
  const mesh={baseSplatTree:{},getSplatTree(){return null;}};
  await assert.rejects(waitForSplatTree(mesh,5,1),/did not finish/);
  mesh.disposed=true;
  await assert.rejects(waitForSplatTree(mesh,5,1),/disposed/);
});

test('an already prepared index needs no extra delay', async () => {
  await waitForSplatTree({baseSplatTree:null,getSplatTree(){return {}; }},0,1);
});
