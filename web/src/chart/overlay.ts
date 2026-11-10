import type {
  IPrimitivePaneRenderer, IPrimitivePaneView, ISeriesApi, ISeriesPrimitive, SeriesAttachedParameter, Time,
} from "lightweight-charts";
import type { CanvasRenderingTarget2D } from "fancy-canvas";
import type { Drawing, DPoint } from "./drawings";
import type { DrawTrade } from "./strategyDraw";

export interface BandPt { time: number; hi: number; lo: number; up: boolean }

/** Converts a bar time (s, may lie outside loaded data) to a media x coordinate. */
export type XOf = (t: number) => number | null;

const C = {
  hold: "rgba(139,124,246,0.055)", holdHatch: "rgba(139,124,246,0.10)", holdLine: "rgba(139,124,246,0.55)",
  holdText: "rgba(178,168,255,0.85)", isText: "rgba(154,163,178,0.55)",
  draw: "#ffb020", drawSel: "#ffd27a", drawFill: "rgba(255,176,32,0.08)",
};

class HoldoutRenderer implements IPrimitivePaneRenderer {
  constructor(private o: OverlayPrimitive) {}
  draw(target: CanvasRenderingTarget2D) {
    const o = this.o;
    if (o.holdout == null || o.hideHoldout || !o.xOf) return;
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      const W = mediaSize.width, H = mediaSize.height;
      let x = o.xOf!(o.holdout!);
      if (x == null) return;
      if (x > W) return;
      x = Math.max(0, Math.round(x)) + 0.5;
      // Plain background (user request): only a thin dashed cutoff line + labels, no fill or hatch.
      ctx.save();
      ctx.strokeStyle = C.holdLine;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 9.5px 'JetBrains Mono Variable', monospace";
      ctx.textBaseline = "bottom";
      if (W - x > 70) { ctx.fillStyle = C.holdText; ctx.fillText("HOLDOUT ▸", x + 6, H - 6); }
      if (x > 80) { ctx.fillStyle = C.isText; ctx.textAlign = "right"; ctx.fillText("◂ IN-SAMPLE", x - 6, H - 6); }
      ctx.restore();
    });
  }
}

/** Indicator band fill (e.g. channel high/low) coloured by regime, behind candles. */
class BandRenderer implements IPrimitivePaneRenderer {
  constructor(private o: OverlayPrimitive) {}
  draw(target: CanvasRenderingTarget2D) {
    const o = this.o, s = o.series, b = o.bands;
    if (!s || !o.xOf || b.length < 2) return;
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      const W = mediaSize.width;
      let prev: { x: number; yh: number; yl: number; up: boolean } | null = null;
      for (const p of b) {
        const x = o.xOf!(p.time);
        if (x == null) { prev = null; continue; }
        const yh = s.priceToCoordinate(p.hi), yl = s.priceToCoordinate(p.lo);
        if (yh == null || yl == null) { prev = null; continue; }
        if (prev && x >= -20 && prev.x <= W + 20) {
          ctx.fillStyle = prev.up ? "rgba(14,168,148,0.10)" : "rgba(242,92,84,0.09)";
          ctx.beginPath(); ctx.moveTo(prev.x, prev.yh); ctx.lineTo(x, yh); ctx.lineTo(x, yl); ctx.lineTo(prev.x, prev.yl); ctx.closePath(); ctx.fill();
        }
        prev = { x, yh, yl, up: p.up };
        if (x > W + 20) break;
      }
    });
  }
}

/** Thin dashed connector from each trade's entry to its exit, green for winners and red for losers. */
class TradeRenderer implements IPrimitivePaneRenderer {
  constructor(private o: OverlayPrimitive) {}
  draw(target: CanvasRenderingTarget2D) {
    const o = this.o, s = o.series;
    if (!s || !o.xOf || !o.trades.length) return;
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      ctx.save();
      ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
      for (const t of o.trades) {
        if (t.exitT == null || t.exitP == null || !isFinite(t.entryP)) continue;
        const x1 = o.xOf!(t.entryT), x2 = o.xOf!(t.exitT);
        if (x1 == null || x2 == null || x2 < 0 || x1 > mediaSize.width) continue;
        const y1 = s.priceToCoordinate(t.entryP), y2 = s.priceToCoordinate(t.exitP);
        if (y1 == null || y2 == null) continue;
        ctx.strokeStyle = t.win ? "rgba(14,168,148,0.9)" : "rgba(242,92,84,0.9)";
        ctx.globalAlpha = t.hold ? 0.55 : 1;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        if (o.focus === t.n) {
          ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.fillStyle = "rgba(255,176,32,0.10)";
          ctx.fillRect(Math.min(x1, x2) - 4, 0, Math.abs(x2 - x1) + 8, mediaSize.height);
          ctx.setLineDash([4, 3]);
        }
      }
      ctx.restore();
    });
  }
}

class DrawRenderer implements IPrimitivePaneRenderer {
  constructor(private o: OverlayPrimitive) {}
  draw(target: CanvasRenderingTarget2D) {
    const o = this.o;
    const list = o.preview ? [...o.drawings, o.preview] : o.drawings;
    if (!list.length || !o.series || !o.xOf) return;
    target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
      for (const d of list) {
        const g = o.geom(d, mediaSize.width);
        if (!g) continue;
        const sel = d.id === o.selected;
        ctx.save();
        ctx.strokeStyle = sel ? C.drawSel : C.draw;
        ctx.lineWidth = sel ? 2 : 1.5;
        if (d === o.preview) ctx.setLineDash([5, 4]);
        ctx.beginPath();
        if (d.kind === "rect") {
          ctx.fillStyle = C.drawFill;
          ctx.fillRect(g.x1, g.y1, g.x2 - g.x1, g.y2 - g.y1);
          ctx.strokeRect(g.x1 + 0.5, g.y1 + 0.5, g.x2 - g.x1, g.y2 - g.y1);
        } else {
          ctx.moveTo(g.x1, g.y1); ctx.lineTo(g.x2, g.y2); ctx.stroke();
        }
        if (d.kind === "hline") {
          ctx.font = "600 10px 'JetBrains Mono Variable', monospace";
          const txt = d.a.p.toLocaleString("pt-BR", { maximumFractionDigits: 2 });
          const w = ctx.measureText(txt).width + 8;
          ctx.fillStyle = sel ? C.drawSel : C.draw;
          ctx.fillRect(mediaSize.width - w - 4, g.y1 - 8, w, 16);
          ctx.fillStyle = "#0a0d12"; ctx.textBaseline = "middle";
          ctx.fillText(txt, mediaSize.width - w, g.y1);
        }
        if (sel || d === o.preview) {
          ctx.fillStyle = "#0a0d12"; ctx.setLineDash([]);
          const pts = d.kind === "hline" ? [] : [[g.x1, g.y1], [g.x2, g.y2]];
          for (const [px, py] of pts) { ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
        }
        ctx.restore();
      }
    });
  }
}

class V implements IPrimitivePaneView {
  constructor(private r: IPrimitivePaneRenderer, private z: "bottom" | "top") {}
  zOrder() { return this.z; }
  renderer() { return this.r; }
}

/** Holdout shading (behind candles) + user drawings (above candles), one primitive per pane. */
export class OverlayPrimitive implements ISeriesPrimitive<Time> {
  series: ISeriesApi<"Candlestick"> | null = null;
  holdout: number | null = null;
  hideHoldout = false;
  drawings: Drawing[] = [];
  preview: Drawing | null = null;
  selected: string | null = null;
  xOf: XOf | null = null;
  private req: (() => void) | null = null;
  bands: BandPt[] = [];
  trades: DrawTrade[] = [];
  focus: number | null = null;
  private views = [new V(new HoldoutRenderer(this), "bottom"), new V(new BandRenderer(this), "bottom"),
    new V(new TradeRenderer(this), "top"), new V(new DrawRenderer(this), "top")];
  attached(p: SeriesAttachedParameter<Time>) { this.series = p.series as ISeriesApi<"Candlestick">; this.req = p.requestUpdate; }
  detached() { this.series = null; this.req = null; }
  paneViews() { return this.views; }
  updateAllViews() {}
  update() { this.req?.(); }

  geom(d: Drawing, W: number) {
    const s = this.series, xOf = this.xOf;
    if (!s || !xOf) return null;
    const y = (p: DPoint) => s.priceToCoordinate(p.p);
    const y1 = y(d.a);
    if (y1 == null) return null;
    if (d.kind === "hline") return { x1: 0, y1, x2: W, y2: y1 };
    if (!d.b) return null;
    const x1 = xOf(d.a.t), x2 = xOf(d.b.t), y2 = y(d.b);
    if (x1 == null || x2 == null || y2 == null) return null;
    if (d.kind === "rect") return { x1: Math.min(x1, x2), y1: Math.min(y1, y2), x2: Math.max(x1, x2), y2: Math.max(y1, y2) };
    return { x1, y1, x2, y2 };
  }

  /** Nearest drawing within `tol` px of (x, y). */
  hit(x: number, y: number, W: number, tol = 6): Drawing | null {
    let best: Drawing | null = null, bd = tol;
    for (const d of this.drawings) {
      const g = this.geom(d, W);
      if (!g) continue;
      let dist: number;
      if (d.kind === "hline") dist = Math.abs(y - g.y1);
      else if (d.kind === "rect") {
        const inside = x >= g.x1 && x <= g.x2 && y >= g.y1 && y <= g.y2;
        const edge = Math.min(Math.abs(x - g.x1), Math.abs(x - g.x2), Math.abs(y - g.y1), Math.abs(y - g.y2));
        dist = inside ? Math.min(edge, tol - 1) : edge + (x < g.x1 - tol || x > g.x2 + tol || y < g.y1 - tol || y > g.y2 + tol ? 99 : 0);
      } else {
        const dx = g.x2 - g.x1, dy = g.y2 - g.y1, L = dx * dx + dy * dy || 1;
        const u = Math.max(0, Math.min(1, ((x - g.x1) * dx + (y - g.y1) * dy) / L));
        dist = Math.hypot(x - (g.x1 + u * dx), y - (g.y1 + u * dy));
      }
      if (dist < bd) { bd = dist; best = d; }
    }
    return best;
  }
}
