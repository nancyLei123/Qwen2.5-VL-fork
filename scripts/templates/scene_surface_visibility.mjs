import {validFigurePose, recordedViewFrameSpec} from './scene_pose_figure.mjs';

export const SURFACE_POLICY = Object.freeze({size:256, near:.30, far:12, alphaLow:.20, alphaHigh:.40,
  depthAgreementBase:.04, depthAgreementSlope:.015, depthMatchBase:.035, depthMatchSlope:.02});

export function acceptsSurfaceDepth(depth, low, high, neighbors = []) {
  const p=SURFACE_POLICY;
  if (![depth,low,high,...neighbors].every(Number.isFinite)
      || depth<p.near || depth>p.far || low<p.near || high<p.near || low>p.far || high>p.far) return false;
  const stable=p.depthAgreementBase+p.depthAgreementSlope*low;
  return Math.abs(low-high)<=stable && Math.abs(depth-low)<=p.depthMatchBase+p.depthMatchSlope*low
    && neighbors.every(v=>v>=p.near && v<=p.far && Math.abs(v-low)<=stable);
}

export function visibilityShader(fragment) {
  const anchor='gl_FragColor = vec4(color.rgb, opacity);';
  if (fragment.split(anchor).length !== 2 || fragment.split('void main () {').length !== 2)
    throw new Error('Unsupported Gaussian renderer: surface overlay left off.');
  const declarations=`
    uniform sampler2D pdDepthLow;
    uniform sampler2D pdDepthHigh;
    uniform mat4 pdRecordedVP;
    uniform mat4 pdObserverInverseVP;
    uniform vec2 pdScreenSize;
    uniform float pdSurfaceEnabled;
    float pdDepth(vec4 packedDepth) {
      if (packedDepth.a < 0.5) return -1.0;
      return dot(packedDepth.rgb, vec3(65536.0,256.0,1.0)) * (255.0*12.0/16777215.0);
    }
    float pdSurfaceWeight() {
      if (pdSurfaceEnabled < 0.5) return 0.0;
      vec2 screenNdc=gl_FragCoord.xy/pdScreenSize*2.0-1.0;
      vec4 world=pdObserverInverseVP*vec4(screenNdc,gl_FragCoord.z*2.0-1.0,1.0);
      world/=world.w;
      vec4 recorded=pdRecordedVP*world;
      if (recorded.w < 0.30 || recorded.w > 12.0) return 0.0;
      vec2 uv=recorded.xy/recorded.w*0.5+0.5;
      if (any(lessThan(uv,vec2(0.01))) || any(greaterThan(uv,vec2(0.99)))) return 0.0;
      float lo=pdDepth(texture2D(pdDepthLow,uv));
      float hi=pdDepth(texture2D(pdDepthHigh,uv));
      float stable=0.04+0.015*lo;
      if (lo < 0.30 || hi < 0.30 || abs(lo-hi)>stable) return 0.0;
      float matchTolerance=0.035+0.02*lo;
      if (abs(recorded.w-lo)>matchTolerance) return 0.0;
      vec2 e=vec2(1.0/256.0,0.0);
      float a=pdDepth(texture2D(pdDepthLow,uv+e.xy));
      float b=pdDepth(texture2D(pdDepthLow,uv-e.xy));
      float c=pdDepth(texture2D(pdDepthLow,uv+e.yx));
      float d=pdDepth(texture2D(pdDepthLow,uv-e.yx));
      if (min(min(a,b),min(c,d))<0.30 || max(max(abs(a-lo),abs(b-lo)),max(abs(c-lo),abs(d-lo)))>stable) return 0.0;
      return 1.0-smoothstep(matchTolerance*0.65,matchTolerance,abs(recorded.w-lo));
    }
  `;
  return fragment.replace('void main () {',declarations+'\nvoid main () {')
    .replace(anchor,`float pdWeight=pdSurfaceWeight();
      color=mix(color,vec3(0.06,0.95,0.78),pdWeight*0.55);
      gl_FragColor = vec4(color.rgb, opacity);`);
}

function depthShaders(vertex) {
  const anchor='vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);';
  if (vertex.split(anchor).length !== 2 || !vertex.includes('varying vec4 vColor;'))
    throw new Error('Unsupported depth shader: surface overlay left off.');
  return {vertex:vertex.replace('varying vec4 vColor;','varying vec4 vColor;\nvarying float pdViewDepth;')
    .replace(anchor,anchor+'\npdViewDepth=-viewCenter.z;'),fragment:`
    precision highp float;
    varying vec4 vColor;
    varying vec2 vPosition;
    varying float pdViewDepth;
    uniform float pdAlphaCutoff;
    void main() {
      float a=dot(vPosition,vPosition);
      if (a>8.0 || exp(-0.5*a)*vColor.a<pdAlphaCutoff || pdViewDepth<0.30 || pdViewDepth>12.0) discard;
      float packed=floor(pdViewDepth/12.0*16777215.0+0.5);
      vec3 rgb=mod(floor(packed/vec3(65536.0,256.0,1.0)),256.0)/255.0;
      gl_FragColor=vec4(rgb,1.0);
    }`};
}

export class GaussianSurfaceVisibility {
  constructor(THREE, viewer, data, elements) {
    this.T=THREE;this.viewer=viewer;this.data=data;this.el=elements;
    this.camera=new THREE.PerspectiveCamera(90,1,.05,15);
    this.camera.up.set(0,0,1);
    this.depthScene=new THREE.Scene();this.key='';this.failed=false;
    this.size=new THREE.Vector2();this.inverseVP=new THREE.Matrix4();this.recordedVP=new THREE.Matrix4();
    this.pixels=new Uint8Array(SURFACE_POLICY.size**2*4);
    this.lastSample=-Infinity;
    const debug=viewer.renderer.debug, previous=debug.onShaderError;
    debug.onShaderError=(gl,program,vertex,fragment)=>{
      const code=gl.getShaderSource(fragment)||'';
      if(code.includes('pdSurfaceWeight') || code.includes('pdAlphaCutoff')) {
        console.warn('Surface shader compilation:',gl.getProgramInfoLog(program),gl.getShaderInfoLog(fragment));
        this.fail(new Error('This GPU could not compile the optional surface overlay.'));
      } else if(previous)previous(gl,program,vertex,fragment);
      else console.error('Scene shader compilation:',gl.getProgramInfoLog(program));
    };
  }

  install(alignment) {
    if (this.data.trial_id !== 'RoomScan3c' || alignment.trial_id !== this.data.trial_id
        || alignment.model_sha256 !== this.data.provenance.model_sha256
        || alignment.graph_uid !== this.data.provenance.graph_uid)
      throw new Error('Surface overlay identity mismatch');
    this.alignment=alignment;
  }

  attach() {
    const T=this.T, source=this.viewer.splatMesh;
    if (this.source===source && source.material===this.tint) return;
    if (this.source) throw new Error('Model changed: surface overlay must be revalidated.');
    if (source.dynamicMode || source.getSceneCount() !== 1) throw new Error('Surface prototype requires one static validated room model.');
    const material=source.material, shaders=depthShaders(material.vertexShader);
    const fragment=visibilityShader(material.fragmentShader);
    this.source=source;this.original=material;
    this.targets=[0,1].map(()=>new T.WebGLRenderTarget(SURFACE_POLICY.size,SURFACE_POLICY.size,
      {minFilter:T.NearestFilter,magFilter:T.NearestFilter,depthBuffer:true,stencilBuffer:false}));
    for(const target of this.targets)target.texture.colorSpace=T.NoColorSpace;
    const size=SURFACE_POLICY.size, k=this.alignment.camera;
    const uniforms={...material.uniforms,focal:{value:new T.Vector2(size*k.fx/k.width,size*k.fy/k.height)},
      viewport:{value:new T.Vector2(size,size)},basisViewport:{value:new T.Vector2(1/size,1/size)},
      orthographicMode:{value:0},orthoZoom:{value:1},inverseFocalAdjustment:{value:1},
      sphericalHarmonicsDegree:{value:0},fadeInComplete:{value:1},pdAlphaCutoff:{value:SURFACE_POLICY.alphaLow}};
    this.depthMaterial=new T.ShaderMaterial({uniforms,vertexShader:shaders.vertex,fragmentShader:shaders.fragment,
      defines:{...material.defines},side:T.DoubleSide,depthTest:true,depthWrite:true,transparent:false,blending:T.NoBlending,toneMapped:false});
    // Independent all-splat indices: the observer's culling/sort cannot remove an occluder.
    const geometry=new T.InstancedBufferGeometry().copy(source.geometry);
    const count=source.getSplatCount(), indices=Uint32Array.from({length:count},(_,i)=>i);
    geometry.setAttribute('splatIndex',new T.InstancedBufferAttribute(indices,1,false));
    geometry.instanceCount=count;geometry.setDrawRange(0,6);
    this.depthMesh=new T.Mesh(geometry,this.depthMaterial);this.depthMesh.frustumCulled=false;
    this.depthMesh.matrixAutoUpdate=false;this.depthScene.add(this.depthMesh);
    this.uniforms={...material.uniforms,pdDepthLow:{value:this.targets[0].texture},pdDepthHigh:{value:this.targets[1].texture},
      pdRecordedVP:{value:this.recordedVP},pdObserverInverseVP:{value:this.inverseVP},
      pdScreenSize:{value:this.size},pdSurfaceEnabled:{value:0}};
    this.tint=new T.ShaderMaterial({uniforms:this.uniforms,vertexShader:material.vertexShader,fragmentShader:fragment,
      defines:{...material.defines},transparent:material.transparent,blending:material.blending,
      depthTest:material.depthTest,depthWrite:material.depthWrite,side:material.side,toneMapped:material.toneMapped});
    source.material=this.tint;
  }

  renderDepth(frame, pose) {
    const T=this.T, r=this.viewer.renderer, p=SURFACE_POLICY;
    const s=recordedViewFrameSpec(this.alignment.camera);
    if(!s)throw new Error('Missing calibrated reconstruction camera');
    this.camera.projectionMatrix.makePerspective(.05*s.left,.05*s.right,.05*s.top,.05*s.bottom,.05,15);
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    this.camera.position.fromArray(pose);this.camera.quaternion.fromArray(pose,3).normalize();
    this.camera.updateMatrixWorld(true);
    this.recordedVP.multiplyMatrices(this.camera.projectionMatrix,this.camera.matrixWorldInverse);
    this.source.updateMatrixWorld(true);this.depthMesh.matrix.copy(this.source.matrixWorld);
    const gl=r.getContext();
    const previous={target:r.getRenderTarget(),color:r.getClearColor(new T.Color()),alpha:r.getClearAlpha(),dither:gl.isEnabled(gl.DITHER),
      viewport:r.getViewport(new T.Vector4()),scissor:r.getScissor(new T.Vector4()),scissorTest:r.getScissorTest(),autoClear:r.autoClear};
    try {
      r.autoClear=false;r.setScissorTest(false);r.setClearColor(0x000000,0);gl.disable(gl.DITHER);
      for(let i=0;i<2;i++) {
        this.depthMaterial.uniforms.pdAlphaCutoff.value=i?p.alphaHigh:p.alphaLow;
        r.setRenderTarget(this.targets[i]);r.clear(true,true,false);r.render(this.depthScene,this.camera);
      }
      if(Math.abs(frame-this.lastSample)>=30) {
        r.readRenderTargetPixels(this.targets[1],0,0,p.size,p.size,this.pixels);
        let supported=0;for(let i=3;i<this.pixels.length;i+=4)if(this.pixels[i])supported++;
        this.supported=supported;this.lastSample=frame;
      }
    } finally {
      r.setRenderTarget(previous.target);r.setViewport(previous.viewport);r.setScissor(previous.scissor);
      r.setScissorTest(previous.scissorTest);r.setClearColor(previous.color,previous.alpha);r.autoClear=previous.autoClear;
      if(previous.dither)gl.enable(gl.DITHER);
    }
    this.key=String(frame);
  }

  update(frame,pose,mode,ready,videoReady) {
    if(this.uniforms)this.uniforms.pdSurfaceEnabled.value=0;
    this.el.stage.dataset.surfaceVisible='false';
    this.el.stage.dataset.surfaceFrame='';
    this.el.stage.dataset.surfacePose='';
    this.el.stage.dataset.surfaceDepthSamples='';
    if(mode!=='surface')return;
    if(this.failed){this.el.status.textContent=this.failureMessage;return;}
    if(!this.alignment || !ready || !videoReady || !validFigurePose(pose)
        || !Number.isInteger(frame) || frame<0 || frame>=this.alignment.frame_lut.length || this.alignment.frame_lut[frame]<0) {
      this.el.status.textContent='Surface visibility waits for the matching video, camera pose and room model; no missing surfaces are filled.';
      return;
    }
    try {
      this.attach();
      if(this.key!==String(frame))this.renderDepth(frame,pose);
      if(this.failed)return;
      this.viewer.camera.updateMatrixWorld(true);
      this.inverseVP.multiplyMatrices(this.viewer.camera.projectionMatrix,this.viewer.camera.matrixWorldInverse).invert();
      this.viewer.renderer.getDrawingBufferSize(this.size);
      this.uniforms.pdSurfaceEnabled.value=1;
      this.el.stage.dataset.surfaceVisible='true';this.el.stage.dataset.surfaceFrame=frame;
      this.el.stage.dataset.surfacePose=JSON.stringify(pose);
      this.el.stage.dataset.surfaceDepthSamples=String(this.supported||0);
      this.el.status.textContent='Cyan on reconstructed objects = estimated camera-visible region. Depth disagreement and edges stay uncolored. Uncolored may mean outside the crop, occluded, or unknown. Experimental model visibility, not eye gaze or measured area.';
    } catch(error) {
      this.fail(error);
    }
  }

  fail(error) {
    this.failed=true;
    if(this.uniforms)this.uniforms.pdSurfaceEnabled.value=0;
    if(this.source&&this.original)this.source.material=this.original;
    this.el.stage.dataset.surfaceVisible='false';
    this.failureMessage=`Surface visibility unavailable: ${error.message} Original scene and point frame are unchanged.`;
    this.el.status.textContent=this.failureMessage;
    console.warn('Optional surface visibility:',error);
  }
}
