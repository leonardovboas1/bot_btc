import type {
  IPrimitivePaneRenderer, IPrimitivePaneView, ISeriesApi, ISeriesPrimitive, SeriesAttachedParameter, Time,
} from "lightweight-charts";
import type { CanvasRenderingTarget2D } from "fancy-canvas";

export interface Grid { times: number[]; prices: number[]; a: number[][]; b?: number[][]; colA: string; colB?: string }
export interface Profile { prices: number[]; volume: number[]; poc?: number }
export interface LayerData { book: Grid | null; liq: Grid | null; vp: Profile | null }

const rgba = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a.toFixed(3)})`;
};
const q98 = (g: number[][][]) => {
  const v: number[] = [];
  for (const m of g) for (const r of m) for (const x of r) if (x > 0) v.push(x);
  if (!v.length) return 1;
  v.sort((x, y) => x - y);
  return v[Math.floor(v.length * 0.98)] || v[v.length - 1] || 1;
};

class Renderer implements IPrimitivePaneRenderer {
  constructor(private p: LayersPrimitive) {}
  draw(target: CanvasRenderingTarget2D) {
    const { series, chart, data } = this.p;
    if (!series || !chart || !data) return;
    target.useBitmapCoordinateSpace(({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr, bitmapSize }) => {
      const ts = chart.timeScale();
      const drawGrid = (g: Grid | null) => {
        if (!g || g.times.length < 1 || g.prices.length < 2) return;
        const step = g.times.length > 1 ? g.times[g.times.length - 1] - g.times[g.times.length - 2] : 3600;
        const ys = g.prices.map((p) => series.priceToCoordinate(p));
        const grids = g.b ? [g.a, g.b] : [g.a];
        const mx = q98(grids);
        const cols = [g.colA, g.colB ?? g.colA];
        for (let i = 0; i < g.times.length; i++) {
          const x0 = ts.timeToCoordinate(g.times[i] as unknown as Time);
          const x1 = ts.timeToCoordinate((g.times[i] + step) as unknown as Time);
          if (x0 == null || x1 == null) continue;
          const px = Math.round(x0 * hr), pw = Math.max(1, Math.round((x1 - x0) * hr) + 1);
          for (let k = 0; k < grids.length; k++) {
            const row = grids[k][i];
            for (let j = 0; j < row.length; j++) {
              const v = row[j];
              if (v <= 0) continue;
              const y = ys[j];
              if (y == null) continue;
              const yu = ys[j + 1] ?? (ys[j - 1] != null ? 2 * y - (ys[j - 1] as number) : y + 4);
              const h = Math.max(1, Math.abs(yu - y) * vr + 1);
              ctx.fillStyle = rgba(cols[k], Math.min(1, v / mx) * 0.85);
              ctx.fillRect(px, Math.round(y * vr - h / 2), pw, Math.round(h));
            }
          }
        }
      };
      drawGrid(data.book);
      drawGrid(data.liq);
      const vp = data.vp;
      if (vp && vp.prices.length > 1) {
        const mx = Math.max(...vp.volume) || 1;
        const maxW = bitmapSize.width * 0.2;
        ctx.fillStyle = "rgba(76,141,255,0.35)";
        for (let j = 0; j < vp.prices.length; j++) {
          const y = series.priceToCoordinate(vp.prices[j]);
          const yn = series.priceToCoordinate(vp.prices[Math.min(j + 1, vp.prices.length - 1)]);
          const yp = series.priceToCoordinate(vp.prices[Math.max(j - 1, 0)]);
          if (y == null) continue;
          const h = Math.max(1, Math.abs(((yn ?? y) - (yp ?? y)) / 2) * vr);
          const w = (vp.volume[j] / mx) * maxW;
          ctx.fillRect(bitmapSize.width - w, Math.round(y * vr - h / 2), w, Math.round(h));
        }
        if (vp.poc != null) {
          const y = series.priceToCoordinate(vp.poc);
          if (y != null) {
            ctx.strokeStyle = "#e0b53a"; ctx.lineWidth = Math.max(1, Math.round(vr));
            ctx.setLineDash([6 * hr, 4 * hr]);
            ctx.beginPath(); ctx.moveTo(0, y * vr); ctx.lineTo(bitmapSize.width, y * vr); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = "#e0b53a"; ctx.font = `${Math.round(13 * vr)}px system-ui`;
            ctx.fillText("POC", bitmapSize.width - maxW - 40 * hr, y * vr - 4 * vr);
          }
        }
      }
    });
  }
}

class View implements IPrimitivePaneView {
  private r: Renderer;
  constructor(p: LayersPrimitive) { this.r = new Renderer(p); }
  zOrder() { return "bottom" as const; }
  renderer() { return this.r; }
}

/** Pane primitive drawn behind candles: book/liquidation heatmaps + volume profile. */
export class LayersPrimitive implements ISeriesPrimitive<Time> {
  series: ISeriesApi<"Candlestick"> | null = null;
  chart: SeriesAttachedParameter<Time>["chart"] | null = null;
  data: LayerData | null = null;
  private req: (() => void) | null = null;
  private v = [new View(this)];
  attached(p: SeriesAttachedParameter<Time>) {
    this.chart = p.chart; this.series = p.series as ISeriesApi<"Candlestick">; this.req = p.requestUpdate;
  }
  detached() { this.chart = null; this.series = null; this.req = null; }
  paneViews() { return this.v; }
  updateAllViews() {}
  setData(d: LayerData | null) { this.data = d; this.req?.(); }
}
