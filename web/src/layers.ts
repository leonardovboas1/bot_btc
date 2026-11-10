import type {
  IPrimitivePaneRenderer, IPrimitivePaneView, ISeriesApi, ISeriesPrimitive, SeriesAttachedParameter, Time,
} from "lightweight-charts";
import type { CanvasRenderingTarget2D } from "fancy-canvas";

export interface Grid { times: number[]; prices: number[]; a: number[][]; b?: number[][]; colA: string; colB?: string }
export interface Profile { prices: number[]; volume: number[]; poc?: number }

/**
 * Everything a layer needs to paint one frame. Layers write pixels into `px` (uint32 ABGR, bitmap
 * coordinates) or return an overlay callback that draws with the 2D context afterwards.
 */
export interface Frame {
  W: number; H: number; hr: number; vr: number;
  /** Bar duration (s) of the pane timeframe, when known: lets layers merge sub-bar columns into bar cells. */
  tfSec?: number;
  px: Uint32Array;
  xOf(time: number): number | null;   // bitmap px
  yOf(price: number): number | null;  // bitmap px
}
export type Overlay = (ctx: CanvasRenderingContext2D) => void;

/**
 * HOOK: a heatmap layer plugs into a pane via `LayersPrimitive.setLayer(layer)`.
 * Implement `paint` (runs only when range/size/rev change, never on crosshair moves) and bump `rev`
 * whenever the layer's data changes. Remove with `removeLayer(id)`.
 */
export interface HeatmapLayer {
  id: string;
  rev: number;
  paint(f: Frame): Overlay | void;
}

const hexRgb = (hex: string) => { const n = parseInt(hex.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
const q98 = (g: number[][][]) => {
  const v: number[] = [];
  for (const m of g) for (const r of m) for (const x of r) if (x > 0) v.push(x);
  if (!v.length) return 1;
  v.sort((x, y) => x - y);
  return v[Math.floor(v.length * 0.98)] || v[v.length - 1] || 1;
};

/** Grid heatmap (order book / estimated liquidations), painted with ImageData. */
export class GridLayer implements HeatmapLayer {
  rev = 0;
  private g: Grid | null = null;
  constructor(public id: string) {}
  setData(g: Grid | null) { this.g = g; this.rev++; }
  paint(f: Frame) {
    const g = this.g;
    if (!g || g.times.length < 1 || g.prices.length < 2) return;
    const step = g.times.length > 1 ? g.times[g.times.length - 1] - g.times[g.times.length - 2] : 3600;
    const ys = g.prices.map((p) => f.yOf(p));
    const grids = g.b ? [g.a, g.b] : [g.a];
    const mx = q98(grids);
    const cols = [g.colA, g.colB ?? g.colA].map(hexRgb);
    for (let i = 0; i < g.times.length; i++) {
      const x0 = f.xOf(g.times[i]), x1 = f.xOf(g.times[i] + step);
      if (x0 == null || x1 == null) continue;
      const xa = Math.max(0, Math.round(x0)), xb = Math.min(f.W, Math.max(xa + 1, Math.round(x1) + 1));
      if (xa >= f.W || xb <= 0) continue;
      for (let k = 0; k < grids.length; k++) {
        const row = grids[k][i];
        const [r, gg, b] = cols[k];
        for (let j = 0; j < row.length; j++) {
          const v = row[j];
          if (v <= 0) continue;
          const y = ys[j];
          if (y == null) continue;
          const yu = ys[j + 1] ?? (ys[j - 1] != null ? 2 * y - (ys[j - 1] as number) : y + 4 * f.vr);
          const h = Math.max(1, Math.abs(yu - y) + 1);
          const ya = Math.max(0, Math.round(y - h / 2)), yb = Math.min(f.H, Math.round(y + h / 2) + 1);
          const a = Math.round(Math.min(1, v / mx) * 0.85 * 255);
          const c = ((a << 24) | (b << 16) | (gg << 8) | r) >>> 0;
          for (let yy = ya; yy < yb; yy++) {
            const o = yy * f.W;
            for (let xx = xa; xx < xb; xx++) if ((f.px[o + xx] >>> 24) < a) f.px[o + xx] = c;
          }
        }
      }
    }
  }
}

/** Volume profile on the right edge (+ POC line). */
export class VpLayer implements HeatmapLayer {
  rev = 0;
  private vp: Profile | null = null;
  constructor(public id = "vp") {}
  setData(v: Profile | null) { this.vp = v; this.rev++; }
  paint(f: Frame): Overlay | void {
    const vp = this.vp;
    if (!vp || vp.prices.length < 2) return;
    return (ctx) => {
      const mx = Math.max(...vp.volume) || 1;
      const maxW = f.W * 0.2;
      ctx.fillStyle = "rgba(76,141,255,0.35)";
      for (let j = 0; j < vp.prices.length; j++) {
        const y = f.yOf(vp.prices[j]);
        if (y == null) continue;
        const yn = f.yOf(vp.prices[Math.min(j + 1, vp.prices.length - 1)]);
        const yp = f.yOf(vp.prices[Math.max(j - 1, 0)]);
        const h = Math.max(1, Math.abs(((yn ?? y) - (yp ?? y)) / 2));
        const w = (vp.volume[j] / mx) * maxW;
        ctx.fillRect(f.W - w, Math.round(y - h / 2), w, Math.round(h));
      }
      if (vp.poc != null) {
        const y = f.yOf(vp.poc);
        if (y != null) {
          ctx.strokeStyle = "#e0b53a"; ctx.lineWidth = Math.max(1, Math.round(f.vr));
          ctx.setLineDash([6 * f.hr, 4 * f.hr]);
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(f.W, y); ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#e0b53a"; ctx.font = `${Math.round(12 * f.vr)}px system-ui`;
          ctx.fillText("POC", f.W - maxW - 36 * f.hr, y - 4 * f.vr);
        }
      }
    };
  }
}

class Renderer implements IPrimitivePaneRenderer {
  private cv = document.createElement("canvas");
  private key = "";
  constructor(private p: LayersPrimitive) {}
  draw(target: CanvasRenderingTarget2D) {
    const { series, chart, layers } = this.p;
    if (!series || !chart || !layers.size) return;
    target.useBitmapCoordinateSpace(({ context, horizontalPixelRatio: hr, verticalPixelRatio: vr, bitmapSize }) => {
      const W = bitmapSize.width, H = bitmapSize.height;
      if (!W || !H) return;
      const ts = chart.timeScale();
      const lr = ts.getVisibleLogicalRange();
      const key = [W, H, lr?.from, lr?.to, this.p.tfSec, this.p.rev, series.priceToCoordinate(1000), series.priceToCoordinate(100000),
        ...[...layers.values()].map((l) => l.id + l.rev)].join("|");
      if (key !== this.key) {
        this.key = key;
        this.cv.width = W; this.cv.height = H;
        const c = this.cv.getContext("2d")!;
        const img = c.createImageData(W, H);
        const f: Frame = {
          W, H, hr, vr, px: new Uint32Array(img.data.buffer), tfSec: this.p.tfSec ?? undefined,
          // time -> x via the pane mapping (works for timestamps between bars, e.g. 15m heatmap columns on a 4h chart)
          xOf: (t) => { const x = this.p.timeToX ? this.p.timeToX(t) : ts.timeToCoordinate(t as unknown as Time); return x == null ? null : x * hr; },
          yOf: (pr) => { const y = series.priceToCoordinate(pr); return y == null ? null : y * vr; },
        };
        const overlays: Overlay[] = [];
        for (const l of layers.values()) { const o = l.paint(f); if (o) overlays.push(o); }
        c.putImageData(img, 0, 0);
        overlays.forEach((o) => o(c));
      }
      context.drawImage(this.cv, 0, 0);
    });
  }
}

class View implements IPrimitivePaneView {
  private r: Renderer;
  constructor(p: LayersPrimitive) { this.r = new Renderer(p); }
  zOrder() { return "bottom" as const; }
  renderer() { return this.r; }
}

/** Pane primitive drawn behind candles; hosts any number of HeatmapLayers. */
export class LayersPrimitive implements ISeriesPrimitive<Time> {
  series: ISeriesApi<"Candlestick"> | null = null;
  chart: SeriesAttachedParameter<Time>["chart"] | null = null;
  layers = new Map<string, HeatmapLayer>();
  /** Optional time -> media x mapping supplied by the pane (extrapolates between/after bars). */
  timeToX: ((t: number) => number | null) | null = null;
  tfSec: number | null = null;
  private req: (() => void) | null = null;
  private v = [new View(this)];
  attached(p: SeriesAttachedParameter<Time>) {
    this.chart = p.chart; this.series = p.series as ISeriesApi<"Candlestick">; this.req = p.requestUpdate;
  }
  detached() { this.chart = null; this.series = null; this.req = null; }
  paneViews() { return this.v; }
  updateAllViews() {}
  /** Bump when the pane bars change (x mapping changes even if the visible logical range does not). */
  rev = 0;
  invalidate() { this.rev++; this.req?.(); }
  setLayer(l: HeatmapLayer) { this.layers.set(l.id, l); this.req?.(); }
  removeLayer(id: string) { if (this.layers.delete(id)) this.req?.(); }
}
