import test from 'node:test';
import assert from 'node:assert/strict';
import {validateSpatialMap, spatialBounds, cameraWire, routeArrays, routePickTime} from '../scripts/templates/scene_spatial_overview.mjs';
import {routeSegments} from '../scripts/templates/scene_route_navigation.mjs';

test('accepts only finite metric landmarks in the same trial and world frame', () => {
  const data = {trial_id:'A',provenance:{graph_uid:'world'}};
  const map = {trial_id:'A',graph_uid:'world',units:'meters',up_axis:'Z',points:[[1,2,3]]};
  assert.deepEqual(validateSpatialMap(map,data),map.points);
  for(const change of [{trial_id:'B'},{graph_uid:'other'},{units:'mm'},{up_axis:'Y'},{points:[]},{points:[[NaN,0,0]]}])
    assert.throws(()=>validateSpatialMap({...map,...change},data));
});
test('overview bounds contain every landmark, with a nondegenerate stationary range', () => {
  assert.deepEqual(spatialBounds([[1,2,3],[1,2,3]]),[[.75,1.25],[1.75,2.25],[2.75,3.25]]);
  const points = [[-80,2,-10],[90,4,7]]; const ranges=spatialBounds(points);
  for(const p of points) p.forEach((v,j)=>assert.ok(v>ranges[j][0] && v<ranges[j][1]));
});
test('frustum uses calibrated quaternion with negative Z forward and does not mutate poses', () => {
  const pose=[1,2,3,0,0,0,1], before=[...pose]; const wire=cameraWire(pose,1);
  assert.deepEqual(wire.corners,[[0,1,2],[2,1,2],[2,3,2],[0,3,2]]);
  assert.deepEqual(wire.forward,[[1,2,3],[1,2,1.4]]);
  assert.equal(wire.edges.length,24); assert.deepEqual(pose,before);
  const rotated=cameraWire([0,0,0,0,Math.SQRT1_2,0,Math.SQRT1_2],1);
  assert.ok(Math.abs(rotated.forward[1][0]+1.6)<1e-12);
  assert.ok(Math.abs(rotated.forward[1][2])<1e-12);
  assert.deepEqual(cameraWire(null).edges,[]);
});
test('route plot preserves video timestamps and breaks at pose gaps', () => {
  const poses=Array.from({length:12},(_,i)=>[i/10,0,1,0,0,0,1]); poses[5]=null;
  const edges=routeSegments({poses,fps:10},0,1.2,.2), arrays=routeArrays(edges);
  assert.ok(edges.every(e=>!(e.a.time<.5 && e.b.time>.5)));
  assert.equal(arrays.x.length,edges.length*3);
  edges.forEach((e,i)=>{
    assert.equal(arrays.customdata[i*3],e.a.time);
    assert.equal(arrays.customdata[i*3+1],e.b.time);
    assert.equal(arrays.x[i*3+2],null);
  });
});
test('3D route click supports Plotly customdata and its trace-index fallback', () => {
  assert.equal(routePickTime({curveNumber:1,customdata:0}),0);
  assert.equal(routePickTime({curveNumber:2,pointNumber:1,data:{customdata:[10,10.2,null]}}),10.2);
  assert.equal(routePickTime({curveNumber:0,customdata:10}),null);
  assert.equal(routePickTime({curveNumber:4,customdata:10}),null);
  assert.equal(routePickTime({curveNumber:1,pointNumber:2,data:{customdata:[10,10.2,null]}}),null);
});
