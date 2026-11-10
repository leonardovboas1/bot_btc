export const fmtTime = (t: number) =>
  new Date(t * 1000).toLocaleString("pt-BR", { timeZone: "UTC", dateStyle: "short", timeStyle: "short" }) + " UTC";
