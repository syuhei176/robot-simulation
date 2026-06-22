import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type {
  StairDynamicsFrame,
  StairDynamicsReplay,
  StairFrameDiagnostics,
} from '../sim3d/stair-dynamics.ts';
import type { QuadBodyLayout, QuadFrame } from '../sim3d/quadruped-dynamics.ts';
import { COURSES, type CourseSpec } from '../sim3d/course.ts';

const LINK_COLOR = 0x16a3b8;
const HEAD_COLOR = 0xf2a33a;
const EDGE_COLOR = 0xe8f5f7;
const STAIR_COLOR = 0x4c5560;
const SUPPORT_COLOR = 0x2dd4bf;
const SLIP_COLOR = 0xfacc15;
const TORQUE_COLOR = 0xfb923c;
const FALL_COLOR = 0xef4444;
const DEFAULT_VIEW_MORPHOLOGY = {
  n: 8,
  totalLength: 0.9,
  bodyWidth: 0.04,
  bodyThickness: 0.028,
};

export class StairDynamicsView {
  readonly scene = new THREE.Scene();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly links: THREE.Group[] = [];
  private readonly linkMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly outlineMaterials: THREE.LineBasicMaterial[] = [];
  private readonly stairs = new THREE.Group();
  private readonly quadGroup = new THREE.Group();
  private readonly quadMeshes: THREE.Mesh[] = [];
  private readonly quadMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly quadJoints: Array<{
    mesh: THREE.Mesh;
    material: THREE.MeshStandardMaterial;
    bodyIdx: number;
    lz: number;
    foot: boolean;
  }> = [];
  // 3D 蛇（MuJoCo）。z-up→y-up の group にカプセルを並べる。移動が分かるよう軌跡(trail)＋距離目盛り＋開始マーカーを置く。
  private readonly snake3dGroup = new THREE.Group();
  private readonly snake3dMeshes: THREE.Mesh[] = [];
  private readonly snake3dMaterials: THREE.MeshStandardMaterial[] = [];
  private readonly snake3dDecor = new THREE.Group(); // trail / 目盛り / 開始マーカー（group 配下＝z-up）
  private snake3dHeadDot: THREE.Mesh | null = null;
  private readonly tmpVec = new THREE.Vector3();
  private readonly tmpQuat = new THREE.Quaternion();
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

    this.scene.add(this.quadGroup);
    this.quadGroup.visible = false;
    this.snake3dGroup.rotation.x = -Math.PI / 2; // sim z-up → three y-up
    this.snake3dGroup.add(this.snake3dDecor);
    this.scene.add(this.snake3dGroup);
    this.snake3dGroup.visible = false;

    this.buildEnvironment();
    this.buildCourse();
    this.buildLinks(DEFAULT_VIEW_MORPHOLOGY.n);

    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  setVisible(visible: boolean): void {
    this.renderer.domElement.style.display = visible ? 'block' : 'none';
  }

  setReplay(replay: StairDynamicsReplay): void {
    this.replay = replay;
    this.buildCourse(replay.summary.config.course);
    this.buildLinks(replay.summary.config.morphology.n);
    this.applyFrame(replay.frames[0]);
    this.controls.target.set(-0.14, 0.16, 0);
    this.camera.position.set(0.32, 0.5, 0.82);
    this.resize();
  }

  /** 蛇(2D階段) / 四足 / 3D蛇(MuJoCo) の表示を切り替える。 */
  setMechanism(mechanism: 'snake' | 'quad' | 'snake3d'): void {
    const snake2d = mechanism === 'snake';
    const quad = mechanism === 'quad';
    const snake3d = mechanism === 'snake3d';
    for (const link of this.links) link.visible = snake2d;
    this.stairs.visible = snake2d || quad;
    this.quadGroup.visible = quad;
    this.snake3dGroup.visible = snake3d;
    if (quad) {
      this.controls.target.set(0.05, 0.11, 0);
      this.camera.position.set(0.18, 0.34, 0.66);
    } else if (snake3d) {
      this.stairs.visible = false;
      this.controls.target.set(0, 0.05, 0);
      this.camera.position.set(0.1, 0.5, 0.9);
    }
  }

  /**
   * 3D 蛇（MuJoCo）のカプセルリンクを layout から構築する。group は z-up→y-up 済みなので
   * 子はシム座標(p,q)のまま置ける。カプセルは three では Y 軸長手なので Z まわり 90° で X 長手にする。
   */
  buildSnake3D(layout: Array<{ half: [number, number, number] }>): void {
    this.snake3dGroup.clear(); // decor 含め全消去 → decor を付け直す
    this.snake3dDecor.clear();
    this.snake3dGroup.add(this.snake3dDecor);
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

    // 距離目盛り（x 方向 0.25m 毎の横線）。軌跡の x 範囲を覆う。
    let minX = Infinity;
    let maxX = -Infinity;
    for (const p of headPts) {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
    }
    const tickMat = new THREE.LineBasicMaterial({
      color: 0x4c5560,
      transparent: true,
      opacity: 0.6,
    });
    const lo = Math.floor(minX / 0.25) * 0.25;
    const hi = Math.ceil(maxX / 0.25) * 0.25;
    for (let x = lo; x <= hi + 1e-9; x += 0.25) {
      const tick = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, -0.15, 0.001),
          new THREE.Vector3(x, 0.15, 0.001),
        ]),
        tickMat,
      );
      this.snake3dDecor.add(tick);
    }

    // 頭の現在位置ドット（毎フレーム更新）。
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

  /**
   * 四足など course を持つ機構用に地形メッシュを構築して表示する。
   * 地形（this.stairs）は高さ=three-y=sim-z で四足リプレイと整合するので、そのまま重ねて描ける。
   * 長いコース（合成コース等）は端まで見えるようカメラを引く。
   */
  showCourse(course: CourseSpec): void {
    this.buildCourse(course);
    this.stairs.visible = true;
    const span = course.goalX;
    if (span > 1.2) {
      // 端から端まで俯瞰できる位置にカメラを引く。
      this.controls.target.set(span * 0.5, 0.12, 0);
      this.camera.position.set(span * 0.5, 0.55, Math.max(1.1, span * 0.9));
    }
  }

  /**
   * 四足の動的リプレイ用メッシュを layout から構築する。シミュは z-up なので
   * quadGroup を x まわり -90° 回転して three の y-up へ変換し、子は sim 座標のまま置く。
   */
  buildQuadReplay(layout: QuadBodyLayout[]): void {
    this.quadGroup.clear();
    this.quadMeshes.length = 0;
    this.quadMaterials.length = 0;
    this.quadJoints.length = 0;
    this.quadGroup.rotation.x = -Math.PI / 2;

    for (const item of layout) {
      const isTrunk = item.kind === 'trunk';
      const material = new THREE.MeshStandardMaterial({
        color: isTrunk ? 0xb9c6d2 : LINK_COLOR,
        roughness: 0.45,
        metalness: isTrunk ? 0.2 : 0.34,
        emissive: isTrunk ? 0x10141a : 0x002a31,
        emissiveIntensity: 0.2,
      });
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(item.half[0] * 2, item.half[1] * 2, item.half[2] * 2),
        material,
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.quadGroup.add(mesh);
      this.quadMeshes.push(mesh);
      this.quadMaterials.push(material);
    }

    // 関節マーカー: layout は [trunk, thigh,shin, thigh,shin, ...]。
    // 各脚に hip（thigh上端）・膝（thigh下端）・足首（shin下端）の球を置く。
    for (let li = 0; li * 2 + 2 < layout.length; li++) {
      const thighIdx = 1 + li * 2;
      const shinIdx = 2 + li * 2;
      const thighHalfZ = layout[thighIdx].half[2];
      const shinHalfZ = layout[shinIdx].half[2];
      this.addQuadJoint(thighIdx, +thighHalfZ, false); // hip
      this.addQuadJoint(thighIdx, -thighHalfZ, false); // 膝
      this.addQuadJoint(shinIdx, -shinHalfZ, true); // 足首
    }
  }

  /** 関節マーカー球を1つ追加（body の局所 z=lz の位置に毎フレーム置く）。 */
  private addQuadJoint(bodyIdx: number, lz: number, foot: boolean): void {
    const material = new THREE.MeshStandardMaterial({
      color: foot ? 0x0c5563 : HEAD_COLOR,
      roughness: 0.4,
      metalness: 0.4,
      emissive: foot ? 0x041d23 : 0x3a1e00,
      emissiveIntensity: 0.4,
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(foot ? 0.013 : 0.016, 16, 12), material);
    mesh.castShadow = true;
    this.quadGroup.add(mesh);
    this.quadJoints.push({ mesh, material, bodyIdx, lz, foot });
  }

  /** 1フレームの剛体姿勢（sim座標 p,q）を適用。転倒中は赤、通常は teal。 */
  applyQuadFrame(frame: QuadFrame): void {
    for (let i = 0; i < this.quadMeshes.length; i++) {
      const b = frame.bodies[i];
      if (!b) continue;
      this.quadMeshes[i].position.set(b.p[0], b.p[1], b.p[2]);
      this.quadMeshes[i].quaternion.set(b.q[0], b.q[1], b.q[2], b.q[3]);
    }
    for (let i = 0; i < this.quadMaterials.length; i++) {
      const base = i === 0 ? 0xb9c6d2 : LINK_COLOR;
      this.quadMaterials[i].color.setHex(frame.diag.fallen ? FALL_COLOR : base);
    }
    // 関節マーカー: 対応する剛体の局所 z=lz をワールドへ変換して配置
    for (const joint of this.quadJoints) {
      const b = frame.bodies[joint.bodyIdx];
      if (!b) continue;
      this.tmpQuat.set(b.q[0], b.q[1], b.q[2], b.q[3]);
      this.tmpVec.set(0, 0, joint.lz).applyQuaternion(this.tmpQuat);
      joint.mesh.position.set(
        b.p[0] + this.tmpVec.x,
        b.p[1] + this.tmpVec.y,
        b.p[2] + this.tmpVec.z,
      );
      joint.material.color.setHex(
        frame.diag.fallen ? FALL_COLOR : joint.foot ? 0x0c5563 : HEAD_COLOR,
      );
    }
    // trunk の sim x を滑らかに追従（group の x 回転では three x と一致）
    const tx = frame.bodies[0]?.p[0] ?? 0;
    this.followTarget.set(tx, 0.1, 0);
    this.desiredCamera.set(tx + 0.06, 0.34, 0.78);
    this.controls.target.lerp(this.followTarget, 0.1);
    this.camera.position.lerp(this.desiredCamera, 0.1);
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

  /** コース（地形）の箱を描画する。コライダと同じ CourseSpec.boxes を使うので両者が一致する。 */
  private buildCourse(course: CourseSpec = COURSES.stairs()): void {
    this.stairs.clear();
    for (const box of course.boxes) {
      // 高さ 0 の箱（平地の薄板など）は描画してもグリッドと干渉するだけなので、底のある箱のみ。
      if (box.halfZ <= 1e-6) continue;
      this.addStairBlock(box.cx, box.cz, box.halfX * 2, box.halfZ * 2);
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
    this.linkMaterials.length = 0;
    this.outlineMaterials.length = 0;

    const m = this.replay?.summary.config.morphology ?? DEFAULT_VIEW_MORPHOLOGY;
    const linkLen = m.totalLength / m.n;
    const body = new THREE.BoxGeometry(linkLen, m.bodyThickness, m.bodyWidth);
    const edge = new THREE.EdgesGeometry(body);

    for (let i = 0; i < count; i++) {
      const group = new THREE.Group();
      const isHead = i === count - 1;
      const material = new THREE.MeshStandardMaterial({
        color: isHead ? HEAD_COLOR : LINK_COLOR,
        roughness: 0.44,
        metalness: 0.32,
        emissive: isHead ? 0x3a1e00 : 0x002a31,
        emissiveIntensity: isHead ? 0.4 : 0.2,
      });
      const mesh = new THREE.Mesh(body, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);

      const outlineMaterial = new THREE.LineBasicMaterial({
        color: EDGE_COLOR,
        transparent: true,
        opacity: isHead ? 0.9 : 0.55,
      });
      const outline = new THREE.LineSegments(edge, outlineMaterial);
      group.add(outline);

      this.scene.add(group);
      this.links.push(group);
      this.linkMaterials.push(material);
      this.outlineMaterials.push(outlineMaterial);
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
    this.applyDiagnostics(frame.diagnostics);
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

    this.applyDiagnostics(u < 0.5 ? a.diagnostics : b.diagnostics);
  }

  private applyDiagnostics(diagnostics?: StairFrameDiagnostics): void {
    const supported = new Set(diagnostics?.supportedLinks ?? []);
    const slipping = new Set(diagnostics?.slippingLinks ?? []);
    const falling = new Set(diagnostics?.fallingLinks ?? []);
    const torqueFailed = diagnostics ? diagnostics.torqueRatio > 1 : false;
    const slipFailed = diagnostics?.failureKinds.includes('slip') ?? false;

    for (let i = 0; i < this.linkMaterials.length; i++) {
      const isHead = i === this.linkMaterials.length - 1;
      let color = isHead ? HEAD_COLOR : LINK_COLOR;
      let emissive = isHead ? 0x3a1e00 : 0x002a31;
      let emissiveIntensity = isHead ? 0.4 : 0.2;

      if (falling.has(i)) {
        color = FALL_COLOR;
        emissive = 0x4c0505;
        emissiveIntensity = 0.75;
      } else if (torqueFailed && diagnostics && i > diagnostics.torqueJoint) {
        color = TORQUE_COLOR;
        emissive = 0x4a1c00;
        emissiveIntensity = 0.7;
      } else if (slipping.has(i) || (slipFailed && supported.has(i))) {
        color = SLIP_COLOR;
        emissive = 0x403500;
        emissiveIntensity = 0.55;
      } else if (supported.has(i)) {
        color = SUPPORT_COLOR;
        emissive = 0x003c36;
        emissiveIntensity = 0.38;
      }

      this.linkMaterials[i].color.setHex(color);
      this.linkMaterials[i].emissive.setHex(emissive);
      this.linkMaterials[i].emissiveIntensity = emissiveIntensity;
      this.outlineMaterials[i].color.setHex(falling.has(i) ? 0xffffff : EDGE_COLOR);
      this.outlineMaterials[i].opacity = falling.has(i) ? 0.95 : isHead ? 0.9 : 0.55;
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
