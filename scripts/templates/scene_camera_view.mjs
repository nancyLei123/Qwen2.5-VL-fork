// OrbitControls owns the moving wearer-centered rig; click-to-look owns only
// a viewing offset. Keeping these separate prevents each playback tick from
// undoing a click or moving the orbit pivot onto a wall in the reconstruction.
export class AnchoredCameraView {
  constructor(camera, controls) {
    this.camera = camera;
    this.orbitCamera = camera.clone();
    controls.object = this.orbitCamera;
    this.lookOffset = camera.quaternion.clone().identity();
    this.aimCamera = camera.clone();
    this.locked = false;
  }

  apply() {
    const {camera, orbitCamera} = this;
    camera.position.copy(orbitCamera.position);
    camera.up.copy(orbitCamera.up);
    camera.quaternion.copy(orbitCamera.quaternion).multiply(this.lookOffset);
    camera.fov = orbitCamera.fov;
    camera.aspect = orbitCamera.aspect;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  }

  lookAtScreen(x, y) {
    if (![x, y].every(Number.isFinite) || Math.abs(x) > 1 || Math.abs(y) > 1) return false;
    this.apply();
    const point = this.camera.position.clone().set(x, y, .5).unproject(this.camera);
    this.aimCamera.position.copy(this.camera.position);
    this.aimCamera.up.copy(this.orbitCamera.up);
    this.aimCamera.lookAt(point);
    this.lookOffset.copy(this.orbitCamera.quaternion).invert().multiply(this.aimCamera.quaternion);
    this.locked = true;
    this.apply();
    return true;
  }

  reset() {
    this.lookOffset.identity();
    this.locked = false;
    this.apply();
  }
}

// Distinguish a deliberate tap from orbit/route drags, long presses, cancelled
// gestures, right-clicks and multi-touch. Never seek the video from a view tap.
export class ViewTap {
  down(event) {
    this.cancel();
    if (event.button === 0 && event.isPrimary !== false) {
      this.start = {id: event.pointerId, x: event.clientX, y: event.clientY, time: event.timeStamp};
    }
  }

  move(event) {
    if (this.start?.id === event.pointerId &&
        Math.hypot(event.clientX - this.start.x, event.clientY - this.start.y) > 4) this.cancel();
  }

  up(event) {
    this.move(event);
    const start = this.start;
    this.cancel();
    return !!start && start.id === event.pointerId && event.button === 0 && event.timeStamp - start.time < 500;
  }

  cancel() { this.start = null; }
}
