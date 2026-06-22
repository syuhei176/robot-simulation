import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const LINK_COLOR = 0x16a3b8;
const HEAD_COLOR = 0xf2a33a;

/**
 * MuJoCo 3D 蛇（snake3d）専用の Three.js ビュー。z-up→y-up 変換した group にカプセルリンクを並べ、
 * 地形箱・頭の軌跡(trail)・距離グリッドを重ねて描く。再生は記録済みフレーム(p,q)を毎フレーム適用し、
 * 頭（COM）を上空から追従する。
 */
export class StairDynamicsView {
  readonly scene = new THREE.Scene();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  // 3D 蛇（MuJoCo）。z-up→y-up の group にカプセルを並べる。移動が分かるよう軌跡(trail)＋距離目盛り＋開始マーカーを置く。
  private readonly snake3dGroup = new THREE.Group();
  private readonly snake3dMeshes: THREE.Mesh[] = [];
  private readonly snake3dMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly snake3dDecor = new THREE.Group(); // trail / 目盛り / 開始マーカー（group 配下＝z-up）
  private readonly snake3dTerrain = new THREE.Group(); // コースの地形箱（group 配下＝z-up）
  private snake3dHeadDot: THREE.Mesh | null = null;
  private readonly followTarget = new THREE.Vector3();
  private readonly desiredCamera = new THREE.Vector3();

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
    this.camera.position.set(0.1, 0.5, 0.9);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.target.set(0, 0.05, 0);

    this.snake3dGroup.rotation.x = -Math.PI / 2; // sim z-up → three y-up
    this.snake3dGroup.add(this.snake3dDecor);
    this.snake3dGroup.add(this.snake3dTerrain);
    this.scene.add(this.snake3dGroup);

    this.buildEnvironment();

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  setVisible(visible: boolean): void {
    this.renderer.domElement.style.display = visible ? 'block' : 'none';
  }

  /** 蛇3D の group を表示しカメラを蛇用に置く（再生開始時に呼ぶ）。 */
  showSnake3D(): void {
    this.snake3dGroup.visible = true;
    this.controls.target.set(0, 0.05, 0);
    this.camera.position.set(0.1, 0.5, 0.9);
  }

  /**
   * 3D 蛇（MuJoCo）のカプセルリンクを layout から構築する。group は z-up→y-up 済みなので
   * 子はシム座標(p,q)のまま置ける。カプセルは three では Y 軸長手なので Z まわり 90° で X 長手にする。
   */
  buildSnake3D(layout: Array<{ half: [number, number, number] }>): void {
    this.snake3dGroup.clear(); // decor / terrain 含め全消去 → 付け直す
    this.snake3dDecor.clear();
    this.snake3dGroup.add(this.snake3dDecor);
    this.snake3dGroup.add(this.snake3dTerrain);
    this.snake3dMeshes.length = 0;
    this.snake3dMaterials.length = 0;
    this.snake3dHeadDot = null;
    for (let i = 0; i < layout.length; i++) {
      const half = layout[i].half;
      const radius = half[1];
      const cylLen = Math.max(0.001, half[0] * 2 - radius * 2);
      const geom = new THREE.CapsuleGeometry(radius, cylLen, 6, 12);
      geom.rotateZ(Math.PI / 2); // 長手を local x へ
      const isHead = i === layout.length - 1;
      const material = new THREE.MeshStandardMaterial({
        color: isHead ? HEAD_COLOR : LINK_COLOR,
        roughness: 0.42,
        metalness: 0.34,
        emissive: isHead ? 0x3a1e00 : 0x002a31,
        emissiveIntensity: isHead ? 0.4 : 0.2,
      });
      const mesh = new THREE.Mesh(geom, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.snake3dGroup.add(mesh);
      this.snake3dMeshes.push(mesh);
      this.snake3dMaterials.push(material);
    }
  }

  /**
   * コースの地形（小障害物→階段→踊り場/壁）を箱メッシュで描く。物理は横(y)に広い壁だが、
   * 描画は障害物として見えるよう y 幅を表示用に詰める（snake の進路を塞ぐ段として読める）。
   */
  setSnake3DTerrain(
    boxes: Array<{
      cx: number;
      cy: number;
      cz: number;
      halfX: number;
      halfY: number;
      halfZ: number;
    }>,
  ): void {
    this.snake3dTerrain.clear();
    for (const b of boxes) {
      const dispHalfY = Math.min(b.halfY, 0.45); // 物理の全幅壁は描画では障害物幅に詰める
      const geom = new THREE.BoxGeometry(b.halfX * 2, dispHalfY * 2, b.halfZ * 2);
      const isOverhang = b.cz - b.halfZ > 0.12; // テーブル天板（頭上の張り出し）
      const mesh = new THREE.Mesh(
        geom,
        new THREE.MeshStandardMaterial({
          color: isOverhang ? 0x5a4636 : 0x42505e,
          roughness: 0.85,
          metalness: 0.05,
          transparent: isOverhang,
          opacity: isOverhang ? 0.6 : 1,
        }),
      );
      mesh.position.set(b.cx, b.cy, b.cz);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.snake3dTerrain.add(mesh);
    }
  }

  /**
   * 移動が一目で分かるよう、頭の全軌跡(trail)・開始マーカー・床の距離目盛り(0.25m毎)を置く。
   * 引数は各フレームの頭位置（シム座標 [x,y,z]）。group 配下なので z-up のまま。
   */
  setSnake3DTrail(headPts: Array<[number, number, number]>): void {
    this.snake3dDecor.clear();
    this.snake3dHeadDot = null;
    if (headPts.length === 0) return;

    // 軌跡ライン（床すれすれ）。
    const pts = headPts.map((p) => new THREE.Vector3(p[0], p[1], 0.002));
    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x35c8ff, transparent: true, opacity: 0.85 }),
    );
    this.snake3dDecor.add(trail);

    // 開始マーカー（リング）。
    const start = new THREE.Mesh(
      new THREE.RingGeometry(0.012, 0.022, 20),
      new THREE.MeshBasicMaterial({ color: 0xf2a33a, side: THREE.DoubleSide }),
    );
    start.position.set(headPts[0][0], headPts[0][1], 0.003);
    this.snake3dDecor.add(start);

    // 距離グリッド（0.25m 毎の格子）。サイドワインドは斜めに進むので x/y 両方向に線を引き、
    // どの向きの移動も読めるようにする。軌跡の bbox（少し余白）を覆う。
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of headPts) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    }
    const tickMat = new THREE.LineBasicMaterial({
      color: 0x4c5560,
      transparent: true,
      opacity: 0.6,
    });
    const G = 0.25;
    const gx0 = Math.floor((minX - G) / G) * G;
    const gx1 = Math.ceil((maxX + G) / G) * G;
    const gy0 = Math.floor((minY - G) / G) * G;
    const gy1 = Math.ceil((maxY + G) / G) * G;
    const addLine = (a: THREE.Vector3, b: THREE.Vector3): void => {
      this.snake3dDecor.add(
        new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), tickMat),
      );
    };
    for (let x = gx0; x <= gx1 + 1e-9; x += G) {
      addLine(new THREE.Vector3(x, gy0, 0.001), new THREE.Vector3(x, gy1, 0.001));
    }
    for (let y = gy0; y <= gy1 + 1e-9; y += G) {
      addLine(new THREE.Vector3(gx0, y, 0.001), new THREE.Vector3(gx1, y, 0.001));
    }

    // 頭の現在位置ドット（毎フレーム更新）。
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 12, 10),
      new THREE.MeshBasicMaterial({ color: 0x35c8ff }),
    );
    this.snake3dDecor.add(dot);
    this.snake3dHeadDot = dot;
  }

  /**
   * ライブ駆動用の静的な距離グリッド（0.25m 毎）＋開始マーカー＋頭ドットを decor に置く。
   * 録画再生の setSnake3DTrail と違い、ライブは経路を先に持たない（毎ステップ進む）ので、進行範囲を
   * 覆う固定グリッドを張る。カメラが蛇を追うと、グリッド上を蛇が進む様子で移動・操舵が一目で読める。
   * buildSnake3D が decor を作り直すので、それより後に呼ぶこと。
   */
  setSnake3DLiveGrid(xRange: [number, number], yRange: [number, number]): void {
    this.snake3dDecor.clear();
    this.snake3dHeadDot = null;
    const tickMat = new THREE.LineBasicMaterial({
      color: 0x4c5560,
      transparent: true,
      opacity: 0.6,
    });
    const G = 0.25;
    const [x0, x1] = xRange;
    const [y0, y1] = yRange;
    const addLine = (a: THREE.Vector3, b: THREE.Vector3): void => {
      this.snake3dDecor.add(
        new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), tickMat),
      );
    };
    for (let x = Math.floor(x0 / G) * G; x <= x1 + 1e-9; x += G) {
      addLine(new THREE.Vector3(x, y0, 0.001), new THREE.Vector3(x, y1, 0.001));
    }
    for (let y = Math.floor(y0 / G) * G; y <= y1 + 1e-9; y += G) {
      addLine(new THREE.Vector3(x0, y, 0.001), new THREE.Vector3(x1, y, 0.001));
    }

    // 開始マーカー（リング）。
    const start = new THREE.Mesh(
      new THREE.RingGeometry(0.012, 0.022, 20),
      new THREE.MeshBasicMaterial({ color: 0xf2a33a, side: THREE.DoubleSide }),
    );
    start.position.set(0, 0, 0.003);
    this.snake3dDecor.add(start);

    // 頭の現在位置ドット（applySnake3DFrame で毎フレーム更新）。
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 12, 10),
      new THREE.MeshBasicMaterial({ color: 0x35c8ff }),
    );
    this.snake3dDecor.add(dot);
    this.snake3dHeadDot = dot;
  }

  /** 3D 蛇の1フレーム（各リンクのシム座標 p,q）を適用し、頭を上空から追従する（移動が見えるよう少し引く）。 */
  applySnake3DFrame(frame: {
    bodies: Array<{ p: [number, number, number]; q: [number, number, number, number] }>;
  }): void {
    let cx = 0;
    let cy = 0;
    const meshN = this.snake3dMeshes.length;
    for (let i = 0; i < meshN; i++) {
      const b = frame.bodies[i];
      if (!b) continue;
      this.snake3dMeshes[i].position.set(b.p[0], b.p[1], b.p[2]);
      this.snake3dMeshes[i].quaternion.set(b.q[0], b.q[1], b.q[2], b.q[3]);
      cx += b.p[0];
      cy += b.p[1];
    }
    const n = meshN || 1;
    cx /= n;
    cy /= n;
    const head = frame.bodies[meshN - 1];
    if (this.snake3dHeadDot && head) this.snake3dHeadDot.position.set(head.p[0], head.p[1], 0.004);
    // sim(x,y,z)→three(x, z, -y)。少し上空＆後方から俯瞰して軌跡が見えるようにする。
    this.followTarget.set(cx, 0.02, -cy);
    this.desiredCamera.set(cx + 0.1, 0.85, -cy + 1.0);
    this.controls.target.lerp(this.followTarget, 0.08);
    this.camera.position.lerp(this.desiredCamera, 0.08);
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
  }

  private resize(): void {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
}
