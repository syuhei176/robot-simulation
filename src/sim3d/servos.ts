/**
 * サーボカタログ（単一の真実）。motor-sizing / stair-dynamics / viewer が共有する。
 *
 * stallNm は概算ストールトルク（定格電圧）。feedback=true は位置帰還を持つ
 * （シリアルバス＋エンコーダ）= クローズドループ登攀制御に向く。プラギア・PWMのみの
 * 機種は動的ピーク・発熱で不利。価格は1個あたりの目安 [JPY]。
 */
export interface Servo {
  id: string;
  name: string;
  stallNm: number; // 概算ストールトルク [N·m]
  massG: number; // 質量 [g]
  priceJpy: number; // 1個あたり目安 [JPY]
  feedback: boolean; // 位置帰還（シリアルバス＋エンコーダ）の有無
  gear: 'plastic' | 'metal';
  iface: 'PWM' | 'serial-bus';
  note?: string;
}

const NM_PER_KGCM = 0.0980665; // 1 kg·cm = 0.0980665 N·m
const kgcm = (v: number): number => v * NM_PER_KGCM;

/**
 * サーボカタログ。ビューア/比較で切り替える対象は SELECTABLE_SERVO_IDS が定義する
 * （SG90 / MG90S / SCS0009 / MG996R / STS3215）。残りは静的サイズ逆算スイープの比較用。
 */
export const SERVOS: Servo[] = [
  {
    id: 'sg90',
    name: 'SG90 (6V)',
    stallNm: kgcm(1.8),
    massG: 9,
    priceJpy: 300,
    feedback: false,
    gear: 'plastic',
    iface: 'PWM',
    note: 'プラギア・FBなし。最軽量だが動的ピーク/発熱に弱い',
  },
  {
    id: 'mg90s',
    name: 'MG90S (6V)',
    stallNm: kgcm(2.2),
    massG: 13,
    priceJpy: 500,
    feedback: false,
    gear: 'metal',
    iface: 'PWM',
    note: '金属ギアでSG90より耐久。FBなし',
  },
  {
    id: 'scs0009',
    name: 'Feetech SCS0009 (6V)',
    stallNm: kgcm(2.3),
    massG: 13,
    priceJpy: 700,
    feedback: true,
    gear: 'metal',
    iface: 'serial-bus',
    note: 'シリアルバス＋エンコーダFB。MG90S級トルクで位置帰還あり=制御向き',
  },
  {
    id: 'sts3215',
    name: 'Feetech STS3215 (12V)',
    stallNm: kgcm(30),
    massG: 60,
    priceJpy: 2500,
    feedback: true,
    gear: 'metal',
    iface: 'serial-bus',
    note: '高トルク・FBあり。重い機体や1kg級向け',
  },
  // --- 以下は静的サイズ逆算スイープの比較用（選択UIの主対象ではない） ---
  {
    id: 'xl330',
    name: 'Dynamixel XL330-M288',
    stallNm: 0.4,
    massG: 18,
    priceJpy: 3500,
    feedback: true,
    gear: 'plastic',
    iface: 'serial-bus',
  },
  {
    id: 'mg996r',
    name: 'MG996R (6V)',
    stallNm: 1.0,
    massG: 55,
    priceJpy: 600,
    feedback: false,
    gear: 'metal',
    iface: 'PWM',
    note: '中間サイズ（10kg·cm）。安servo三兄弟(~2)とSTS3215(30)の間を埋める。安価で平地歩行に十分',
  },
  {
    id: 'xl430',
    name: 'Dynamixel XL430-W250',
    stallNm: 1.5,
    massG: 57,
    priceJpy: 7000,
    feedback: true,
    gear: 'metal',
    iface: 'serial-bus',
  },
  {
    id: 'xm430',
    name: 'Dynamixel XM430-W350',
    stallNm: 3.5,
    massG: 82,
    priceJpy: 25000,
    feedback: true,
    gear: 'metal',
    iface: 'serial-bus',
  },
];

/** ユーザーが切り替える主対象のID（ストール昇順。MG996Rが中間サイズ）。 */
export const SELECTABLE_SERVO_IDS = ['sg90', 'mg90s', 'scs0009', 'mg996r', 'sts3215'] as const;

export function getServo(id: string): Servo {
  const servo = SERVOS.find((s) => s.id === id);
  if (!servo) throw new Error(`unknown servo id: ${id}`);
  return servo;
}

/** 要求トルク [N·m] を満たす最小ストールのサーボ（無ければ null）。 */
export function pickServo(requiredNm: number): Servo | null {
  return (
    SERVOS.filter((s) => s.stallNm >= requiredNm).sort((a, b) => a.stallNm - b.stallNm)[0] ?? null
  );
}
