import {loadFont as loadNext} from '@remotion/google-fonts/AtkinsonHyperlegibleNext';
import {loadFont as loadMono} from '@remotion/google-fonts/AtkinsonHyperlegibleMono';
import {AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame} from 'remotion';
import type {Clip, Token} from '../../data';
import {easeOut, pct, useWave} from '../../shared/audio';
import {OVERLAP} from '../../shared/timeline';

// Atkinson Hyperlegible was drawn by the Braille Institute for readers with low vision.
export const {fontFamily: SANS} = loadNext('normal', {weights: ['400', '600', '700', '800'], subsets: ['latin']});
export const {fontFamily: MONO} = loadMono('normal', {weights: ['400', '600'], subsets: ['latin']});

export const L = {
	ground: '#ffffff',
	ink: '#1b1b1b',
	muted: '#646464',
	grid: '#dcdcdc',
	faint: '#efefef',
	cursor: '#c62828',
	wrongFill: '#f9dcd7',
	clear: '#2357c6',
	small: '#8c8c8c',
	large: '#1b1b1b',
};

export const SYSTEMS: {key: string; label: string; color: string}[] = [
	{key: 'whisper-small', label: 'whisper-small', color: L.small},
	{key: 'whisper-large-v3', label: 'whisper-large-v3', color: L.large},
	{key: 'Clear Voice', label: 'Clear Voice', color: L.clear},
];

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
export const PX = 360; // left edge of plots
export const PW = 1460; // plot width

// Plain white sheet; a new figure wipes in from the left over the previous one.
export const Sheet: React.FC<{children: React.ReactNode; first?: boolean}> = ({children, first}) => {
	const f = useCurrentFrame();
	const p = first ? 1 : easeOut(f / OVERLAP);
	return (
		<AbsoluteFill style={{backgroundColor: L.ground, fontFamily: SANS, color: L.ink, clipPath: `inset(0 ${(1 - p) * 100}% 0 0)`}}>
			{children}
		</AbsoluteFill>
	);
};

export const Rise: React.FC<{at: number; children: React.ReactNode; style?: React.CSSProperties}> = ({at, children, style}) => {
	const f = useCurrentFrame();
	const t = easeOut((f - at) / 14);
	return <div style={{opacity: t, transform: `translateY(${(1 - t) * 14}px)`, ...style}}>{children}</div>;
};

export const Caption: React.FC<{n: number; children: React.ReactNode; at?: number; top?: number}> = ({n, children, at = 0, top = 968}) => (
	<Rise at={at} style={{position: 'absolute', top, left: 100, right: 100, fontSize: 24, lineHeight: 1.4, color: L.muted}}>
		<b style={{color: L.ink}}>Figure {n}.</b> {children}
	</Rise>
);

// A word tier: the words a system heard, spread across the plot width; misheard words are tinted.
const Tier: React.FC<{label: string; color: string; tokens: Token[]; top: number; at: number; wer?: number; bold?: boolean}> = ({label, color, tokens, top, at, wer, bold}) => {
	const f = useCurrentFrame();
	const total = tokens.reduce((a, t) => a + t.w.length + 2, 0) || 1;
	let x = 0;
	return (
		<div style={{position: 'absolute', top, left: 100, width: PX - 100 + PW + 30, height: 66}}>
			<div style={{position: 'absolute', left: 0, top: 18, width: PX - 120, fontSize: 26, fontWeight: bold ? 800 : 600, color}}>{label}</div>
			<div style={{position: 'absolute', left: PX - 100, top: 0, width: PW, height: 66, borderTop: `1.5px solid ${L.grid}`}}>
				{tokens.length === 0 && f > at ? <div style={{position: 'absolute', left: 12, top: 16, fontSize: 26, fontStyle: 'italic', color: L.muted}}>(nothing)</div> : null}
				{tokens.map((t, i) => {
					const w = ((t.w.length + 2) / total) * PW;
					const left = x;
					x += w;
					const o = interpolate(f - at - i * 3, [0, 8], [0, 1], clamp);
					const bad = t.tag !== 'ok';
					return (
						<div key={i} style={{position: 'absolute', left, width: w, top: 6, height: 54, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: o}}>
							{bad ? <div style={{position: 'absolute', inset: '4px 3px', background: L.wrongFill}} /> : null}
							<span style={{position: 'relative', fontSize: 30, fontWeight: bold ? 700 : 400, color: bad ? L.cursor : L.ink}}>{t.w}</span>
						</div>
					);
				})}
			</div>
			{wer !== undefined ? (
				<div style={{position: 'absolute', left: PX - 100 + PW + 14, top: 18, fontFamily: MONO, fontSize: 22, color: L.muted, opacity: interpolate(f - at - 20, [0, 10], [0, 1], clamp)}}>
					{pct(wer)}
				</div>
			) : null}
		</div>
	);
};

// The analysis window: waveform, spectrogram and one word tier per system, with a cursor that follows the sound.
export const Analysis: React.FC<{clip: Clip; playAt: number; heading: string; tiersAt: number}> = ({clip, playAt, heading, tiersAt}) => {
	const {bars, progress} = useWave(clip.file, clip.duration, playAt, 400);
	const waveH = 110;
	const top = 150;
	const specTop = top + waveH + 14;
	const specH = 250;
	const axisTop = specTop + specH;
	const tierTop = axisTop + 52;
	const path = bars.length
		? `M0 ${waveH / 2} ` +
			bars.map((a, i) => `L${(i / (bars.length - 1)) * PW} ${waveH / 2 - (a * waveH) / 2}`).join(' ') +
			' ' +
			bars
				.slice()
				.reverse()
				.map((a, i) => `L${((bars.length - 1 - i) / (bars.length - 1)) * PW} ${waveH / 2 + (a * waveH) / 2}`)
				.join(' ') +
			' Z'
		: '';
	const ticks = Array.from({length: Math.floor(clip.duration / 0.5) + 1}, (_, i) => i * 0.5);
	return (
		<>
			<Rise at={2} style={{position: 'absolute', top: 64, left: 100, fontSize: 40, fontWeight: 800}}>
				{heading}
			</Rise>
			<div style={{position: 'absolute', top, left: 100, width: PX - 120, fontSize: 22, color: L.muted, lineHeight: 1.3}}>Sound</div>
			<div style={{position: 'absolute', top: specTop + 6, left: 100, width: PX - 120, fontSize: 22, color: L.muted, lineHeight: 1.3}}>
				Spectrogram
				<br />0 to 6 kHz
			</div>
			<svg style={{position: 'absolute', top, left: PX}} width={PW} height={waveH}>
				<rect width={PW} height={waveH} fill={L.faint} />
				<path d={path} fill="#3a3a3a" />
			</svg>
			<div style={{position: 'absolute', top: specTop, left: PX, width: PW, height: specH, background: L.faint, overflow: 'hidden'}}>
				{clip.spec ? <Img src={staticFile(clip.spec)} style={{width: PW, height: specH}} /> : null}
			</div>
			<svg style={{position: 'absolute', top: axisTop, left: PX, overflow: 'visible'}} width={PW} height={40}>
				{ticks.map((t) => (
					<g key={t}>
						<line x1={(t / clip.duration) * PW} x2={(t / clip.duration) * PW} y1={0} y2={8} stroke={L.muted} strokeWidth={1.5} />
						<text x={(t / clip.duration) * PW} y={32} textAnchor="middle" fontFamily={MONO} fontSize={20} fill={L.muted}>
							{t.toFixed(1)} s
						</text>
					</g>
				))}
			</svg>
			<Tier label="Said" color={L.ink} tokens={clip.ref.split(' ').map((w) => ({w, tag: 'ok' as const}))} top={tierTop} at={tiersAt} />
			{SYSTEMS.map((s, i) =>
				clip.systems[s.key] ? (
					<Tier key={s.key} label={s.label} color={s.color} tokens={clip.systems[s.key].tokens} wer={clip.systems[s.key].wer} top={tierTop + 74 * (i + 1)} at={tiersAt + 24 + i * 22} bold={s.key === 'Clear Voice'} />
				) : null,
			)}
			{progress > 0 && progress < 1 ? (
				<div style={{position: 'absolute', top: top - 8, left: PX + progress * PW - 1, width: 2, height: axisTop - top + 16, background: L.cursor}} />
			) : null}
			{progress > 0 && progress < 1 ? (
				<div style={{position: 'absolute', top: top - 40, left: PX + progress * PW - 60, width: 120, textAlign: 'center', fontFamily: MONO, fontSize: 20, color: L.cursor}}>
					{(progress * clip.duration).toFixed(2)} s
				</div>
			) : null}
		</>
	);
};
