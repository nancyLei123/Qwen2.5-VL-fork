import test from 'node:test';
import assert from 'node:assert/strict';
import {routeSegments, nearestRouteTime, routeWindow, nearbyRouteWindow, moveOrbitAnchor, playbackSceneModel} from '../scripts/templates/scene_route_navigation.mjs';

test('route samples preserve timestamps and break at every unsupported frame', () => {
  const data = {fps:10, poses:Array.from({length:10},(_,i)=>[i*.1,0,0,0,0,0,1])};
  data.poses[3] = null;
  const segments = routeSegments(data,0,1);
  assert.ok(segments.length > 1);
  assert.ok(segments.every(s=>!(s.a.time<.3 && s.b.time>.3)));
  for(const s of segments) assert.ok(Math.abs(s.b.position[0]-s.b.time)<1e-12);
});
test('discontinuous camera jumps are not drawn as a walk', () => {
  const data = {fps:5, poses:[[0,0,0],[.1,0,0],[10,0,0],[10.1,0,0]]};
  const segments = routeSegments(data,0,1);
  assert.ok(!segments.some(s=>s.a.position[0]<1 && s.b.position[0]>9));
});
test('path click uses spatial position to select corresponding video time', () => {
  const s = [{a:[0,0],b:[100,0],start:40,end:60}];
  assert.equal(nearestRouteTime(s,[25,3],0),45);
  assert.equal(nearestRouteTime(s,[25,100],0),null);
});
test('repeated visits prefer current time without replacing the exact time slider', () => {
  const s=[{a:[0,0],b:[100,0],start:0,end:10},{a:[0,0],b:[100,0],start:100,end:110}];
  assert.equal(nearestRouteTime(s,[50,0],103),105);
  assert.equal(nearestRouteTime(s,[50,0],2),5);
});
test('standing still has no invented path length and keeps a nearby timestamp', () => {
  assert.equal(nearestRouteTime([{a:[5,5],b:[5,5],start:0,end:10}],[5,5],7),7);
});
test('route scopes preserve original video clock and session bounds', () => {
  const data={duration_s:3400};
  assert.deepEqual(routeWindow(data,null,'first15',0),[0,900]);
  assert.deepEqual(routeWindow(data,null,'first30',0),[0,1800]);
  assert.deepEqual(routeWindow({duration_s:236.8},null,'first30',0),[0,236.8]);
  assert.deepEqual(routeWindow(data,{video_start_s:530,video_end_s:570},'scene',545),[530,570]);
  assert.deepEqual(routeWindow(data,null,'scene',3399),[3360,3400]);
});

test('nearby route remains short and contains the playhead across session bounds', () => {
  for (const duration of [10, 236.8, 3400]) for (let t=0;t<duration;t+=.7) {
    const [a,b] = nearbyRouteWindow(duration,t);
    assert.ok(a>=0 && b<=duration && a<=t && t<b);
    assert.ok(b-a<=30);
    assert.deepEqual(routeWindow({duration_s:duration},null,'nearby',t),[a,b]);
  }
  assert.deepEqual(nearbyRouteWindow(3400,500),[485,515]);
});
test('camera orbit follows translation without changing orbit angle or distance', () => {
  const old=[5,2,4], camera=[5.3,1.2,4.1], next=[-20,14,-2];
  const translated=moveOrbitAnchor(camera,old,next);
  for(let i=0;i<3;i++) assert.ok(Math.abs((translated[i]-next[i])-(camera[i]-old[i]))<1e-12);
  assert.deepEqual(moveOrbitAnchor(camera,old,old),camera);
});
test('joint display joins submap boundaries but cannot invent additional coverage', () => {
  const a={id:'room'}, b={id:'corridor'}, joint={id:'joint',video_start_s:0,video_end_s:900};
  assert.equal(playbackSceneModel(a,joint,'joint',489),joint);
  assert.equal(playbackSceneModel(b,joint,'joint',490),joint);
  assert.equal(playbackSceneModel(b,joint,'submaps',490),b);
  assert.equal(playbackSceneModel(b,joint,'joint',900),b);
  assert.equal(playbackSceneModel(null,joint,'joint',10),null);
  assert.equal(playbackSceneModel(a,null,'joint',10),a);
  assert.equal(playbackSceneModel(a,joint,'joint30',489),joint);
  assert.equal(playbackSceneModel(b,joint,'joint30',900),b);
  assert.equal(playbackSceneModel(null,joint,'joint30',10),null);
  assert.equal(playbackSceneModel(a,joint,'unreviewed',10),a);
});
