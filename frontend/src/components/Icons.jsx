// Bo icon nho tu ve (SVG, stroke theo currentColor) - khong phu thuoc thu vien ngoai.
const P = {
  play: <path d="M7 4.5v15l12.5-7.5z" fill="currentColor" stroke="none" />,
  pause: <><rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none" /><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none" /></>,
  stepBack: <><path d="M6 5v14" /><path d="M19 6l-9 6 9 6z" fill="currentColor" /></>,
  stepFwd: <><path d="M18 5v14" /><path d="M5 6l9 6-9 6z" fill="currentColor" /></>,
  back5: <><path d="M11 6l-7 6 7 6z" fill="currentColor" /><path d="M20 6l-7 6 7 6z" fill="currentColor" /></>,
  fwd5: <><path d="M13 6l7 6-7 6z" fill="currentColor" /><path d="M4 6l7 6-7 6z" fill="currentColor" /></>,
  target: <><circle cx="12" cy="12" r="7" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4" /></>,
  box: <><rect x="4" y="4" width="16" height="16" rx="2" strokeDasharray="3 3" /></>,
  trash: <><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></>,
  x: <path d="M6 6l12 12M18 6L6 18" />,
  tag: <><path d="M3 12V4h8l10 10-8 8z" /><circle cx="7.5" cy="8.5" r="1.2" fill="currentColor" /></>,
  upload: <><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v4h16v-4" /></>,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.5" /></>,
  spark: <path d="M12 3l2.2 6.3L21 12l-6.8 2.7L12 21l-2.2-6.3L3 12l6.8-2.7z" fill="currentColor" stroke="none" />,
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  edit: <><path d="M4 20h4L19 9l-4-4L4 16z" /></>,
  back: <path d="M15 5l-7 7 7 7" />,
  film: <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="M10 9.5v5l4.5-2.5z" fill="currentColor" /></>,
};

export default function Icon({ name, size = 18, className = "" }) {
  return (
    <svg className={`icon ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none"
         stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {P[name]}
    </svg>
  );
}

// Bieu tuong cau long: nut cau + cac canh long xoe ra
export function Shuttle({ size = 26 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9.6 16.2L5.2 5.2M12 15.6V3.8M14.4 16.2l4.4-11" />
      <path d="M4.2 5.6c5-2.2 10.6-2.2 15.6 0" />
      <circle cx="12" cy="19" r="2.4" fill="currentColor" />
    </svg>
  );
}

// Duong ke san cau long nhin tu tren xuong (dung lam hoa tiet nen)
export function CourtLines({ className = "" }) {
  return (
    <svg className={className} viewBox="0 0 300 600" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true" preserveAspectRatio="xMidYMid slice">
      <rect x="20" y="20" width="260" height="560" />
      <path d="M46 20v560M254 20v560" />
      <path d="M20 300h260" strokeWidth="4" />
      <path d="M46 240h208M46 360h208" />
      <path d="M150 240V20M150 360v220" />
      <path d="M20 60h260M20 540h260" />
    </svg>
  );
}
