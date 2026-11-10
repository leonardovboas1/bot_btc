import { useEffect, useState } from "react";
import type { Backtest } from "./types";
import { fmtTime } from "./util";

type V = number | null | undefined;
const pct = (v: V) => (v == null ? "-" : (v * 100).toFixed(1) + "%");
const num = (v: V) => (v == null ? "-" : v.toFixed(2));
const CARDS: [string, string, (v: V) => string][] = [
  ["Retorno total", "total_return", pct], ["CAGR", "cagr", pct], ["Sharpe", "sharpe", num], ["Sortino", "sortino", num],
  ["Max drawdown", "max_drawdown", pct], ["Profit factor", "profit_factor", num], ["Win rate", "win_rate", pct],
  ["Nº trades", "n_trades", (v) => (v == null ? "-" : String(v))],
];
const PER = 10;

export default function SidePanel({ bt }: { bt: Backtest | null }) {
  const [page, setPage] = useState(0);
  const trades = bt ? [...bt.trades].reverse() : [];
  const pages = Math.max(1, Math.ceil(trades.length / PER));
  useEffect(() => setPage(0), [bt]);
  return (
    <div className="side">
      <div className="cards">
        {CARDS.map(([l, k, f]) => (
          <div className="card" key={k}><small>{l}</small><div>{f(bt?.metrics[k])}</div></div>
        ))}
      </div>
      <div className="trades">
        <small style={{ color: "var(--mut)" }}>Trades (in-sample, mais recentes primeiro)</small>
        <table>
          <thead><tr><th>Entrada</th><th>Lado</th><th>Saída</th><th>P&amp;L</th></tr></thead>
          <tbody>
            {trades.slice(page * PER, page * PER + PER).map((t, i) => (
              <tr key={i}>
                <td>{fmtTime(t.entry_time).slice(0, 14)}</td>
                <td className={t.side === "LONG" ? "pos" : "neg"}>{t.side}</td>
                <td>{t.exit_reason}</td>
                <td className={t.pnl_pct >= 0 ? "pos" : "neg"}>{pct(t.pnl_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="pager">
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>‹</button>
          <span>{page + 1} / {pages}</span>
          <button disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>›</button>
        </div>
      </div>
    </div>
  );
}
