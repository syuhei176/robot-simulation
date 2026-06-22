/** 機構レジストリ。本プロジェクトは MuJoCo 蛇（snake3d）専用。 */
import type { Mechanism } from './Mechanism.ts';
import { snake3dMechanism } from './snake3d.ts';

export const MECHANISMS: Mechanism[] = [snake3dMechanism];

export function getMechanism(id: string): Mechanism {
  const mech = MECHANISMS.find((m) => m.id === id);
  if (!mech) throw new Error(`unknown mechanism: ${id}`);
  return mech;
}
