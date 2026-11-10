export const ALL_TFS = ["5m", "15m", "1h", "4h", "1d", "1w"] as const;
export const TF_SEC: Record<string, number> = { "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800 };
export const TF_LABEL: Record<string, string> = { "5m": "5m", "15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D", "1w": "1S" };
export const fmtTime = (t: number) =>
  new Date(t * 1000).toLocaleString("pt-BR", { timeZone: "UTC", dateStyle: "short", timeStyle: "short" }) + " UTC";
export const fmtDate = (t: number) => new Date(t * 1000).toLocaleDateString("pt-BR", { timeZone: "UTC", day: "2-digit", month: "short", year: "numeric" });
export const decimals = (p: number) => (p >= 1000 ? 1 : p >= 100 ? 2 : p >= 1 ? 3 : 5);
export const fmtPrice = (p: number | null | undefined, d?: number) =>
  p == null || !isFinite(p) ? "—" : p.toLocaleString("pt-BR", { minimumFractionDigits: d ?? decimals(p), maximumFractionDigits: d ?? decimals(p) });
export const fmtPct = (v: number | null | undefined, signed = true, digits = 1) =>
  v == null || !isFinite(v) ? "—" : (signed && v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v * 100).toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits }) + "%";
export const fmtNum = (v: number | null | undefined, digits = 2) =>
  v == null || !isFinite(v) ? "—" : (v < 0 ? "−" : "") + Math.abs(v).toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
export const fmtVol = (v: number) => (v >= 1e9 ? (v / 1e9).toFixed(2) + "B" : v >= 1e6 ? (v / 1e6).toFixed(2) + "M" : v >= 1e3 ? (v / 1e3).toFixed(1) + "K" : v.toFixed(0));
export const isAbort = (e: unknown) => (e as { name?: string })?.name === "AbortError";
export const base = (s: string) => (s.endsWith("USDT") ? s.slice(0, -4) : s);
