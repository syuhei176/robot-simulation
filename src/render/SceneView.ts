import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { SnakePhysics } from '../sim/SnakePhysics.ts';

const ACCENT = 0x35c8ff;
const GRID_SIZE = 200;
const GRID_DIVISIONS = 200;

/**
 * Three.js による描画。マイクロボットの各節を「黒いキューブ + 青い縁取り発光」で
 * 表現し、物理モデルのノード位置/向きに毎フレーム追従させる（ベイマックス調）。
 * 物理は XY 平面、描画は XZ 平面（y を上）にマップする。
 */
export class SceneView {
  readonly scene = new THREE.Scene();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly modules: THREE.Group[] = [];
  private readonly height: number;
  private readonly com = new THREE.Vector3();
  private readonly prevCom = new THREE.Vector3();
  private followPrimed = false;
  follow = true;
  // 無限スクロールする地面・グリッド（蛇に追従させて前進感を出す）
  private ground!: THREE.Mesh;
  private grid!: THREE.GridHelper;
  private readonly gridCell = GRID_SIZE / GRID_DIVISIONS;

  constructor(container: HTMLElement, nodeCount: number, restLength: number) {
    this.height = restLength * 0.5;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene.background = new THREE.Color(0x05070b);
    this.scene.fog = new THREE.Fog(0x05070b, 14, 42);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
    this.camera.position.set(6, 7, 9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0, 0);

    this.buildEnvironment();
    this.buildMicrobots(nodeCount, restLength);

    window.addEventListener('resize', () => this.resize(container));
    this.resize(container);
  }

  setVisible(visible: boolean): void {
    this.renderer.domElement.style.display = visible ? 'block' : 'none';
  }

  private buildEnvironment(): void {
    const hemi = new THREE.HemisphereLight(0x7fb6ff, 0x0a0d12, 0.7);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(8, 14, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 1;
    key.shadow.camera.far = 60;
    const s = 24;
    key.shadow.camera.left = -s;
    key.shadow.camera.right = s;
    key.shadow.camera.top = s;
    key.shadow.camera.bottom = -s;
    this.scene.add(key);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(GRID_SIZE, GRID_SIZE),
      new THREE.MeshStandardMaterial({ color: 0x0b1018, roughness: 0.95, metalness: 0.0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);
    this.ground = ground;

    const grid = new THREE.GridHelper(GRID_SIZE, GRID_DIVISIONS, 0x3a7fb5, 0x1d3043);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    this.scene.add(grid);
    this.grid = grid;
  }

  private buildMicrobots(nodeCount: number, restLength: number): void {
    const baseSize = restLength * 0.82;
    const geom = new THREE.BoxGeometry(baseSize, baseSize, baseSize);
    const edges = new THREE.EdgesGeometry(geom);

    for (let i = 0; i < nodeCount; i++) {
      const isHead = i === 0;
      const group = new THREE.Group();

      const mat = new THREE.MeshStandardMaterial({
        color: isHead ? 0x141a22 : 0x0e1219,
        metalness: 0.65,
        roughness: 0.35,
        emissive: ACCENT,
        emissiveIntensity: isHead ? 0.18 : 0.06,
      });
      const cube = new THREE.Mesh(geom, mat);
      cube.castShadow = true;
      cube.receiveShadow = true;
      if (isHead) cube.scale.setScalar(1.18);
      group.add(cube);

      const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({
          color: ACCENT,
          transparent: true,
          opacity: isHead ? 0.95 : 0.5,
        }),
      );
      if (isHead) line.scale.setScalar(1.18);
      group.add(line);

      if (isHead) {
        const eye = new THREE.Mesh(
          new THREE.SphereGeometry(baseSize * 0.16, 16, 16),
          new THREE.MeshStandardMaterial({
            color: ACCENT,
            emissive: ACCENT,
            emissiveIntensity: 2.2,
          }),
        );
        eye.position.set(baseSize * 0.5, 0, 0);
        group.add(eye);
      }

      group.position.y = this.height;
      this.scene.add(group);
      this.modules.push(group);
    }
  }

  /** 物理状態を描画へ反映。 */
  sync(sim: SnakePhysics): void {
    for (let i = 0; i < this.modules.length; i++) {
      const g = this.modules[i];
      const x = sim.pos[i * 2];
      const y = sim.pos[i * 2 + 1];
      g.position.set(x, this.height, y);

      // 体軸方向へ yaw を合わせる
      let dx: number;
      let dy: number;
      if (i < sim.nodeCount - 1) {
        dx = sim.pos[i * 2] - sim.pos[(i + 1) * 2];
        dy = sim.pos[i * 2 + 1] - sim.pos[(i + 1) * 2 + 1];
      } else {
        dx = sim.pos[(i - 1) * 2] - sim.pos[i * 2];
        dy = sim.pos[(i - 1) * 2 + 1] - sim.pos[i * 2 + 1];
      }
      g.rotation.y = -Math.atan2(dy, dx);
    }

    const c = sim.centerOfMass();
    this.com.set(c[0], this.height, c[1]);
    if (this.follow && this.followPrimed) {
      const delta = this.com.clone().sub(this.prevCom);
      this.camera.position.add(delta);
      this.controls.target.add(delta);
    }
    this.prevCom.copy(this.com);
    this.followPrimed = true;

    // 地面とグリッドを蛇に追従させて無限スクロール化。
    // 地面はそのまま追従（無地なので継ぎ目なし）、グリッドはセル単位でスナップさせ、
    // 線が後方へ「流れて」見えることで前進している感覚を作る。
    this.ground.position.set(c[0], 0, c[1]);
    this.grid.position.set(
      Math.round(c[0] / this.gridCell) * this.gridCell,
      0,
      Math.round(c[1] / this.gridCell) * this.gridCell,
    );
  }

  /** 追従対象を切り替えたときにカメラが飛ばないようにする。 */
  resetFollow(): void {
    this.followPrimed = false;
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private resize(container: HTMLElement): void {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
