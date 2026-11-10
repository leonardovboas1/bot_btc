import type { ReactNode } from "react";

const S = ({ children, size = 16 }: { children: ReactNode; size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">{children}</svg>
);
export const I = {
  cursor: <S><path d="M3.5 2.5l9 4.6-4 1.2-1.6 3.9z" /></S>,
  trend: <S><path d="M3 13L13 3" /><circle cx="3" cy="13" r="1.4" fill="currentColor" /><circle cx="13" cy="3" r="1.4" fill="currentColor" /></S>,
  hline: <S><path d="M1.5 8h13" /><circle cx="8" cy="8" r="1.4" fill="currentColor" /></S>,
  rect: <S><rect x="2.5" y="4" width="11" height="8" rx="0.5" /></S>,
  erase: <S><path d="M6 13.5h7.5M2.8 9.6l5.6-5.6a1.4 1.4 0 012 0l2 2a1.4 1.4 0 010 2L8 12.4H5.6z" /></S>,
  trash: <S><path d="M2.5 4.5h11M6 4.5V3h4v1.5M4 4.5l.7 9h6.6l.7-9" /></S>,
  search: <S><circle cx="7" cy="7" r="4.5" /><path d="M10.5 10.5L14 14" /></S>,
  l1: <S><rect x="2" y="2.5" width="12" height="11" rx="1.2" /></S>,
  l2: <S><rect x="2" y="2.5" width="12" height="11" rx="1.2" /><path d="M8 2.5v11" /></S>,
  l4: <S><rect x="2" y="2.5" width="12" height="11" rx="1.2" /><path d="M8 2.5v11M2 8h12" /></S>,
  cross: <S><path d="M8 1.5v13M1.5 8h13" /><circle cx="8" cy="8" r="2.2" /></S>,
  link: <S><path d="M6.5 9.5l3-3M7 4.5l1-1a2.8 2.8 0 014 4l-1 1M9 11.5l-1 1a2.8 2.8 0 01-4-4l1-1" /></S>,
  equity: <S><path d="M2 12.5l3.5-4 3 2.5L14 4" /><path d="M10.5 4H14v3.5" /></S>,
  panel: <S><rect x="2" y="2.5" width="12" height="11" rx="1.2" /><path d="M10 2.5v11" /></S>,
  chev: <S size={10}><path d="M4 6l4 4 4-4" /></S>,
  bolt: <S size={12}><path d="M9 1.5L3.5 9H8l-1 5.5L12.5 7H8z" fill="currentColor" stroke="none" /></S>,
};

/** Brand mark: a stylised candle split by a signal line. */
export const Logo = () => (
  <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
    <rect x="1" y="1" width="20" height="20" rx="5" fill="#ffb020" />
    <path d="M7 5v12M15 4v14" stroke="#0a0d12" strokeWidth="1.6" strokeLinecap="round" />
    <rect x="5" y="8" width="4" height="6" rx="1" fill="#0a0d12" />
    <rect x="13" y="6.5" width="4" height="8" rx="1" fill="#0a0d12" />
  </svg>
);
