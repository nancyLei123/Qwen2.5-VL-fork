import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {OBJLoader} from 'three/addons/loaders/OBJLoader.js';
import {MTLLoader} from 'three/addons/loaders/MTLLoader.js';
const status = document.getElementById('status'), root = document.getElementById('viewer');
try {
  const response = await fetch('comparison_results.json');
  if (!response.ok) throw new Error('Comparison metadata unavailable');
  const report = await response.json();
  const label = new URLSearchParams(location.search).get('scene') || 'room';
  if (!Object.hasOwn(report.scenes, label)) throw new Error('Unknown scene');
  const data = report.scenes[label], m = data.validation_frame.T_world_camera;
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#17251f');
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  root.appendChild(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(2*Math.atan(data.camera.height/(2*data.camera.fy))*180/Math.PI, 1, .01, 500);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const position = new THREE.Vector3(m[0][3],m[1][3],m[2][3]);
  const forward = new THREE.Vector3(m[0][2],m[1][2],m[2][2]);
  function reset() {
    camera.position.copy(position);
    camera.up.set(-m[0][1],-m[1][1],-m[2][1]);
    controls.target.copy(position).addScaledVector(forward, 2.5);
    camera.lookAt(controls.target); controls.update();
  }
  reset(); document.getElementById('reset').addEventListener('click', reset);
  const materials = await new MTLLoader().setPath(`${label}/`).loadAsync('model.mtl');
  const object = await new OBJLoader().setMaterials(materials).loadAsync(`${label}/model.obj`, event => {
    status.textContent = `Loading ${data.title.toLowerCase()} mesh: ${(event.loaded/1e6).toFixed(1)} MB`;
  });
  object.traverse(child => {
    if (!child.isMesh) return;
    function unlit(material) {
      if (material.map) material.map.colorSpace = THREE.SRGBColorSpace;
      return new THREE.MeshBasicMaterial({map:material.map, color:material.color, side:THREE.FrontSide});
    }
    child.material = Array.isArray(child.material) ? child.material.map(unlit) : unlit(child.material);
  });
  scene.add(object);
  const observer = new ResizeObserver(() => {
    renderer.setSize(root.clientWidth, root.clientHeight);
    camera.aspect = root.clientWidth/root.clientHeight; camera.updateProjectionMatrix();
  }); observer.observe(root);
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
  status.textContent = `${data.title} / ${data.metashape.faces.toLocaleString()} mesh faces`;
} catch (error) { status.textContent = `Viewer unavailable: ${error.message}`; console.error(error); }
