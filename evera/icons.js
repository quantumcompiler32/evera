// Thin line icons (48px grid, stroke = currentColor).
const P = {
  iron: '<path d="M7 41l3.5-3.5"/><path d="M9.2 35.6l3.2 3.2 11-11-3.2-3.2z"/><path d="M22.6 24.4l2.3-2.3 3.2 3.2-2.3 2.3"/><path d="M26.5 23.6L39 11"/><path d="M39 11l2.4-2.4"/><path d="M8 39c-2.5 2.5-4 4.5-3 6"/>',
  driver: '<path d="M6.5 41.5l3 3 9.5-9.5-3-3z"/><path d="M14.5 33.5l2-2 3 3-2 2"/><path d="M18.8 29.2L36 12"/><path d="M36 12l3.5-5.5L42 9l-5.5 3.5"/>',
  dmm: '<rect x="13" y="5" width="22" height="38" rx="4"/><rect x="17" y="9" width="14" height="9" rx="1.5"/><circle cx="24" cy="27.5" r="5"/><path d="M24 27.5l3-3"/><circle cx="19" cy="37.5" r="1.6"/><circle cx="29" cy="37.5" r="1.6"/>',
  tape: '<path d="M8 14h18a10 10 0 0 1 10 10v0a10 10 0 0 1-10 10H8z"/><circle cx="24" cy="24" r="5.5"/><path d="M36 30h7v-3"/><path d="M8 14v20"/>',
  allen: '<path d="M34 6L15.5 31.5a3 3 0 0 0 .7 4.2l7.3 5.3"/><path d="M37.5 8.5L19 34"/>',
  glue: '<path d="M7 15h24l4.5 5L31 25H20.5l-3 15h-6l3-15H7z"/><path d="M35.5 20H42"/><path d="M20.5 25l1.5 7"/><path d="M3 18h4v4H3z"/>',
  pliers: '<path d="M20 5l2.5 15"/><path d="M28 5l-2.5 15"/><circle cx="24" cy="22" r="2.6"/><path d="M22.4 24.2L16 43"/><path d="M25.6 24.2L32 43"/>',
  tweezers: '<path d="M33 6L18 40"/><path d="M36 8L21 42"/><path d="M33 6l3 2"/><path d="M18 40l3 2"/>',
  cutters: '<path d="M18.5 5c-2.5 6-.5 12 4 14.5"/><path d="M29.5 5c2.5 6 .5 12-4 14.5"/><circle cx="24" cy="21.5" r="2.6"/><path d="M22.4 23.7L16 43"/><path d="M25.6 23.7L32 43"/>',
  calipers: '<path d="M5 13h38v6H5z"/><path d="M9 19v16l4 4V19"/><path d="M21 19v12l-4 4"/><path d="M27 15.5h12"/>',
  hand: '<path d="M15 43h18"/><path d="M24 43V31"/><circle cx="24" cy="29" r="2.4"/><path d="M22.4 27.4L13 18"/><path d="M25.6 27.4L35 18"/><path d="M13 18l-4-1 1-4 4 1"/><path d="M35 18l4-1-1-4-4 1"/><circle cx="35" cy="9" r="3"/>',
  dock: '<path d="M24 6l16 8v20l-16 8-16-8V14z"/><path d="M8 14l16 8 16-8"/><path d="M24 22v20"/><path d="M16 10l16 8"/>',
  mic: '<rect x="18" y="6" width="12" height="22" rx="6"/><path d="M11 23a13 13 0 0 0 26 0"/><path d="M24 36v6"/><path d="M18 42h12"/>',
  gear: '<circle cx="24" cy="24" r="6"/><path d="M24 4v6M24 38v6M4 24h6M38 24h6M9.9 9.9l4.2 4.2M33.9 33.9l4.2 4.2M9.9 38.1l4.2-4.2M33.9 14.1l4.2-4.2"/>',
  gamepad: '<path d="M14 15h20a9 9 0 0 1 9 9v1a8 8 0 0 1-14 5l-2-2h-6l-2 2a8 8 0 0 1-14-5v-1a9 9 0 0 1 9-9z"/><path d="M15 20v8M11 24h8"/><circle cx="31" cy="22" r="1.6"/><circle cx="35" cy="26" r="1.6"/>',
  stop: '<rect x="13" y="13" width="22" height="22" rx="3"/>',
  close: '<path d="M12 12l24 24M36 12L12 36"/>',
  play: '<path d="M17 11l20 13-20 13z"/>',
};

export function icon(name, size = 40, stroke = 1.6) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="${(stroke * 48) / size}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ''}</svg>`;
}
