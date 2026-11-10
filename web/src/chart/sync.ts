/** Cross-pane sync of crosshair time and visible time range. Only the pane under the pointer drives. */
export interface Peer {
  cross(t: number | null): void;
  range(from: number, to: number): void; // times (s), may extend past loaded data
  preset?(k: string): void;               // range-bar presets (1D, 5D, ..., Todos)
}
export class SyncBus {
  peers = new Map<number, Peer>();
  cross = true;
  range = false;
  emitCross(from: number, t: number | null) {
    if (!this.cross) return;
    for (const [i, p] of this.peers) if (i !== from) p.cross(t);
  }
  emitRange(from: number, a: number, b: number) {
    if (!this.range) return;
    for (const [i, p] of this.peers) if (i !== from) p.range(a, b);
  }
}
