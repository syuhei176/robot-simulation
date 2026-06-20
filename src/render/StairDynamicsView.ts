import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { StairDynamicsFrame, StairDynamicsReplay } from '../sim3d/stair-dynamics.ts';

const LINK_COLOR = 0x16a3b8;
const HEAD_COLOR = 0xf2a33a;
const EDGE_COLOR = 0xe8f5f7;
const STAIR_COLOR = 0x4c5560;
const DEFAULT_VIEW_MORPHOLOGY = {
  n: 8,
  totalLength: 0.9,
  bodyWidth: 0.04,
  bodyThickness: 0.028,
};
const DEFAULT_VIEW_STAIR = {
  rise: 0.18,
  treadDepth: 0.25,
  stepCount: 3,
};

export class StairDynamicsView {
  readonly scene = new THREE.Scene();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly links: THREE.Group[] = [];
  private readonly stairs = new THREE.Group();
  private readonly followTarget = new THREE.Vector3();
  private readonly desiredCamera = new THREE.Vector3();
  private replay: StairDynamicsReplay | null = null;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.style.display = 'none';
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x070806);
    this.scene.fog = new THREE.Fog(0x070806, 1.8, 3.8);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
    this.camera.position.set(0.3, 0.48, 0.78);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(-0.16, 0.14, 0);

    this.buildEnvironment();
    this.buildStairs();
    this.buildLinks(DEFAULT_VIEW_MORPHOLOGY.n);

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  setVisible(visible: boolean): void {
    this.renderer.domElement.style.display = visible ? 'block' : 'none';
  }

  setReplay(replay: StairDynamicsReplay): void {
    this.replay = replay;
    this.buildStairs(replay);
    this.buildLinks(replay.summary.config.morphology.n);
    this.applyFrame(replay.frames[0]);
    this.controls.target.set(-0.14, 0.16, 0);
    this.camera.position.set(0.32, 0.5, 0.82);
    this.resize();
  }

  duration(): number {
    return this.replay?.summary.config.duration ?? 0;
  }

  applyTime(time: number): void {
    if (!this.replay || this.replay.frames.length === 0) return;
    const t = ((time % this.duration()) + this.duration()) % this.duration();
    const frames = this.replay.frames;

    let hi = 1;
    while (hi < frames.length && frames[hi].t < t) hi++;
    const a = frames[Math.max(0, hi - 1)];
    const b = frames[Math.min(frames.length - 1, hi)];
    const span = Math.max(1e-9, b.t - a.t);
    const u = THREE.MathUtils.clamp((t - a.t) / span, 0, 1);
    this.applyInterpolatedFrame(a, b, u);
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private buildEnvironment(): void {
    const hemi = new THREE.HemisphereLight(0xdde8f0, 0x1b1d17, 0.7);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(-0.4, 1.4, 0.7);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 5;
    key.shadow.camera.left = -1.2;
    key.shadow.camera.right = 1.2;
    key.shadow.camera.top = 1.2;
    key.shadow.camera.bottom = -1.2;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x88b6ff, 0.5);
    fill.position.set(0.8, 0.4, -0.8);
    this.scene.add(fill);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(2.4, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x171916, roughness: 0.95 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.001;
    ground.receiveShadow = true;
    this.scene.add(ground);

    const grid = new THREE.GridHelper(2.4, 24, 0x69705f, 0x2d3328);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.32;
    this.scene.add(grid);

    this.scene.add(this.stairs);
  }

  private buildStairs(replay?: StairDynamicsReplay): void {
    this.stairs.clear();
    const stair = replay?.summary.config.stair ?? DEFAULT_VIEW_STAIR;

    this.addStairBlock(-0.7, -0.02, 1.4, 0.04);
    for (let i = 0; i < stair.stepCount; i++) {
      const height = (i + 1) * stair.rise;
      const x = i * stair.treadDepth + stair.treadDepth / 2;
      this.addStairBlock(x, height / 2, stair.treadDepth, height);
    }
  }

  private addStairBlock(x: number, y: number, width: number, height: number): void {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(width, height, 0.42),
      new THREE.MeshStandardMaterial({
        color: STAIR_COLOR,
        roughness: 0.82,
        metalness: 0.05,
      }),
    );
    mesh.position.set(x, y, 0);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.stairs.add(mesh);

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0xa8b0a8, transparent: true, opacity: 0.55 }),
    );
    mesh.add(edges);
  }

  private buildLinks(count: number): void {
    for (const link of this.links) {
      this.scene.remove(link);
    }
    this.links.length = 0;

    const m = this.replay?.summary.config.morphology ?? DEFAULT_VIEW_MORPHOLOGY;
    const linkLen = m.totalLength / m.n;
    const body = new THREE.BoxGeometry(linkLen, m.bodyThickness, m.bodyWidth);
    const edge = new THREE.EdgesGeometry(body);

    for (let i = 0; i < count; i++) {
      const group = new THREE.Group();
      const isHead = i === count - 1;
      const mesh = new THREE.Mesh(
        body,
        new THREE.MeshStandardMaterial({
          color: isHead ? HEAD_COLOR : LINK_COLOR,
          roughness: 0.44,
          metalness: 0.32,
          emissive: isHead ? 0x3a1e00 : 0x002a31,
          emissiveIntensity: isHead ? 0.4 : 0.2,
        }),
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);

      const outline = new THREE.LineSegments(
        edge,
        new THREE.LineBasicMaterial({
          color: EDGE_COLOR,
          transparent: true,
          opacity: isHead ? 0.9 : 0.55,
        }),
      );
      group.add(outline);

      this.scene.add(group);
      this.links.push(group);
    }
  }

  private applyFrame(frame?: StairDynamicsFrame): void {
    if (!frame) return;
    for (let i = 0; i < this.links.length; i++) {
      const link = frame.links[i];
      if (!link) continue;
      this.links[i].position.set(link.x, link.z, 0);
      this.links[i].rotation.set(0, 0, link.angle);
    }
  }

  private applyInterpolatedFrame(a: StairDynamicsFrame, b: StairDynamicsFrame, u: number): void {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < this.links.length; i++) {
      const la = a.links[i];
      const lb = b.links[i] ?? la;
      if (!la || !lb) continue;
      const x = THREE.MathUtils.lerp(la.x, lb.x, u);
      const y = THREE.MathUtils.lerp(la.z, lb.z, u);
      this.links[i].position.set(x, y, 0);
      this.links[i].rotation.set(0, 0, THREE.MathUtils.lerp(la.angle, lb.angle, u));
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    if (Number.isFinite(minX)) {
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      this.followTarget.set(cx, cy + 0.06, 0);
      this.desiredCamera.set(cx + 0.42, cy + 0.36, 0.88);
      this.controls.target.lerp(this.followTarget, 0.12);
      this.camera.position.lerp(this.desiredCamera, 0.12);
    }
  }

  private resize(): void {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
