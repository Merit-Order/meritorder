import qrcode from "qrcode-generator";
import { useMemo } from "react";

/** A QR code for `value` (an otpauth:// link for authenticator apps), drawn as SVG squares: no HTML injected, so it fits
 * the console's Content-Security-Policy. Dark on white whatever the theme, as phone cameras read best. */
export function QrCode({ value, size = 200, label }: { value: string; size?: number; label: string }) {
  const { n, path } = useMemo(() => {
    const qr = qrcode(0, "M");
    qr.addData(value);
    qr.make();
    const count = qr.getModuleCount();
    let d = "";
    for (let r = 0; r < count; r++) for (let c = 0; c < count; c++) if (qr.isDark(r, c)) d += `M${c + 4} ${r + 4}h1v1h-1z`;
    return { n: count + 8, path: d };  // a 4-module quiet zone on each side
  }, [value]);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${n} ${n}`} role="img" aria-label={label} shapeRendering="crispEdges"
      style={{ background: "#fff", borderRadius: 6, display: "block" }}>
      <path d={path} fill="#000" />
    </svg>
  );
}
