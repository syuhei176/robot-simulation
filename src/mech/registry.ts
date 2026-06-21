/** 機構レジストリ。新機構はここに足すだけで統合ダッシュボードに現れる。 */
import type { Mechanism } from './Mechanism.ts';
import { snakeMechanism } from './snake.ts';
import { quadMechanism } from './quad.ts';
import { multipedMechanism } from './multiped.ts';
import { urchinMechanism } from './urchin.ts';
import { spokeUrchinMechanism } from './spoke-urchin.ts';

export const MECHANISMS: Mechanism[] = [
  snakeMechanism,
  quadMechanism,
  multipedMechanism,
  urchinMechanism,
  spokeUrchinMechanism,
];

export function getMechanism(id: string): Mechanism {
  const mech = MECHANISMS.find((m) => m.id === id);
  if (!mech) throw new Error(`unknown mechanism: ${id}`);
  return mech;
}
