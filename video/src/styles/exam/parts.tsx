import {loadFont as loadCourier} from '@remotion/google-fonts/CourierPrime';
import {loadFont as loadKalam} from '@remotion/google-fonts/Kalam';
import {loadFont as loadFranklin} from '@remotion/google-fonts/LibreFranklin';
import {AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import type {Token} from '../../data';
import {easeOut, useWave} from '../../shared/audio';
import {OVERLAP} from '../../shared/timeline';

export const {fontFamily: PRINT} = loadFranklin('normal', {weights: ['400', '500', '700', '800'], subsets: ['latin']});
export const {fontFamily: TYPE} = loadCourier('normal', {weights: ['400', '700'], subsets: ['latin']});
export const {fontFamily: HAND} = loadKalam('normal', {weights: ['400', '700'], subsets: ['latin']});

export const E = {
	paper: '#fbfbf9',
	ink: '#161616',
	soft: '#5c5c58',
	rule: '#1a1a1a',
	hair: '#d3d3cd',
	red: '#c41e2a', // the examiner's pen: the only colour that marks
	blue: '#1d3fa8', // Clear Voice writes in blue ink
	pencil: '#6b6b66',
};

// Candidate letters follow the order the cover page lists them.
export const CANDIDATES: {key: string; letter: string; name: string; note: string; ink: string}[] = [
	{key: 'whisper-small', letter: 'A', name: 'whisper-small', note: 'stock', ink: E.pencil},
	{key: 'whisper-large-v3', letter: 'B', name: 'whisper-large-v3', note: 'stock, 6× bigger', ink: E.ink},
	{key: 'Clear Voice', letter: 'C', name: 'Clear Voice', note: 'ours', ink: E.blue},
];

export const COL = 1640; // left edge of the examiner's column
export const LEFT = 150;

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// A printed page: running header, page number, "For examiner's use" column. Slides in over the previous page.
export const Page: React.FC<{n: number; children: React.ReactNode; examinerColumn?: boolean; turnOver?: boolean}> = ({
	n,
	children,
	examinerColumn = true,
	turnOver = true,
}) => {
	const f = useCurrentFrame();
	const slide = n === 1 ? 1 : easeOut(f / OVERLAP);
	const shadow = interpolate(f, [0, OVERLAP, OVERLAP + 10], [0.22, 0.22, 0], clamp);
	return (
		<AbsoluteFill
			style={{
				backgroundColor: E.paper,
				transform: `translateX(${(1 - slide) * 1920}px)`,
				boxShadow: `-24px 0 48px rgba(0,0,0,${shadow})`,
				fontFamily: PRINT,
				color: E.ink,
			}}
		>
			{n > 1 ? (
				<div style={{position: 'absolute', top: 46, left: LEFT, right: 1920 - COL - 120, display: 'flex', justifyContent: 'space-between', fontSize: 22, color: E.soft}}>
					<span>Clear Voice · Paper 1</span>
					<span>Unseen speakers, unseen sentences</span>
				</div>
			) : null}
			<div style={{position: 'absolute', top: 84, left: LEFT, right: 120, height: 1.5, background: E.rule}} />
			{examinerColumn ? (
				<>
					<div style={{position: 'absolute', top: 84, bottom: 110, left: COL, width: 1.5, background: E.rule}} />
					<div style={{position: 'absolute', top: 100, left: COL + 16, width: 150, fontSize: 18, lineHeight: 1.25, color: E.soft}}>For examiner’s use</div>
				</>
			) : null}
			<div style={{position: 'absolute', bottom: 110, left: LEFT, right: 120, height: 1.5, background: E.rule}} />
			<div style={{position: 'absolute', bottom: 58, left: 0, right: 0, textAlign: 'center', fontSize: 24, color: E.soft}}>{n}</div>
			{turnOver ? (
				<div style={{position: 'absolute', bottom: 56, right: 120, fontSize: 24, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10}}>
					Turn over
					<svg width={26} height={18} viewBox="0 0 26 18">
						<path d="M1 9h22M15 1l8 8-8 8" fill="none" stroke={E.ink} strokeWidth={2.4} />
					</svg>
				</div>
			) : null}
			{children}
		</AbsoluteFill>
	);
};

// Question number in the margin, the question in print, its mark allocation flush right.
export const Question: React.FC<{n: number | string; marks?: number; children: React.ReactNode; top: number; at?: number}> = ({n, marks, children, top, at = 0}) => {
	const f = useCurrentFrame();
	const o = interpolate(f - at, [0, 10], [0, 1], clamp);
	return (
		<div style={{position: 'absolute', top, left: LEFT - 90, width: COL - LEFT + 60, display: 'flex', gap: 0, opacity: o, fontSize: 40, lineHeight: 1.3}}>
			<div style={{width: 90, fontWeight: 800}}>{n}</div>
			<div style={{flex: 1, fontWeight: 500}}>{children}</div>
			{marks !== undefined ? <div style={{width: 110, textAlign: 'right', fontWeight: 700}}>[{marks}]</div> : null}
		</div>
	);
};

// The recording printed as a figure: ink fills the bars as the sound plays.
export const WaveFigure: React.FC<{file: string; duration: number; playFrom: number; top: number; caption: string; width?: number; height?: number}> = ({
	file,
	duration,
	playFrom,
	top,
	caption,
	width = 1300,
	height = 150,
}) => {
	const {bars, progress} = useWave(file, duration, playFrom, 150);
	const gap = 3;
	const bw = bars.length ? (width - gap * (bars.length - 1)) / bars.length : 0;
	return (
		<div style={{position: 'absolute', top, left: LEFT + 90 + (COL - LEFT - 90 - width) / 2, width}}>
			<svg width={width} height={height}>
				<line x1={0} x2={width} y1={height / 2} y2={height / 2} stroke={E.hair} strokeWidth={1} />
				{bars.map((a, i) => {
					const h = a * height * 0.92;
					return <rect key={i} x={i * (bw + gap)} y={(height - h) / 2} width={bw} height={h} fill={i / bars.length < progress ? E.ink : '#c9c9c3'} />;
				})}
				{progress > 0 && progress < 1 ? <line x1={progress * width} x2={progress * width} y1={-8} y2={height + 8} stroke={E.red} strokeWidth={2} /> : null}
			</svg>
			<div style={{textAlign: 'center', fontSize: 24, marginTop: 14, color: E.soft}}>{caption}</div>
		</div>
	);
};

const STRIKE = 7; // frames for the pen to cross out one word

// A candidate's typed answer on a ruled line. After typing, the examiner crosses out every wrong word
// and writes the score in the examiner's column.
export const Answer: React.FC<{
	cand: (typeof CANDIDATES)[number];
	tokens: Token[];
	refWords: number;
	top: number;
	at: number;
	cps?: number;
	size?: number;
}> = ({cand, tokens, refWords, top, at, cps = 55, size = 40}) => {
	const f = useCurrentFrame();
	const {fps} = useVideoConfig();
	const text = tokens.map((t) => t.w).join(' ');
	const typed = Math.max(0, Math.min(text.length, Math.floor(((f - at) / fps) * cps)));
	const typedEnd = at + Math.ceil((text.length / cps) * fps);
	const bad = tokens.map((t, i) => (t.tag !== 'ok' ? i : -1)).filter((i) => i >= 0);
	const markStart = typedEnd + 6;
	const scoreAt = markStart + bad.length * (STRIKE - 2) + 6;
	const right = Math.max(0, tokens.filter((t) => t.tag === 'ok').length);
	let shown = 0;
	const o = interpolate(f - at + 8, [0, 8], [0, 1], clamp);
	return (
		<div style={{position: 'absolute', top, left: LEFT, width: COL - LEFT - 30, opacity: o}}>
			<div style={{fontSize: 24, color: E.soft, marginBottom: 6}}>
				Candidate {cand.letter} · <span style={{color: E.ink, fontWeight: 700}}>{cand.name}</span> ({cand.note})
			</div>
			<div style={{position: 'relative', minHeight: size * 1.45, borderBottom: `2px dotted ${E.soft}`, paddingBottom: 6, display: 'flex', flexWrap: 'wrap', gap: `0 ${size * 0.6}px`, fontFamily: TYPE, fontSize: size, color: cand.ink}}>
				{tokens.length === 0 && f > at + 4 ? <span style={{fontFamily: PRINT, fontSize: 28, color: E.soft, fontStyle: 'italic'}}>(no answer)</span> : null}
				{tokens.map((t, i) => {
					const start = shown;
					shown += t.w.length + 1;
					const visible = t.w.slice(0, Math.max(0, typed - start));
					if (!visible) return null;
					const k = bad.indexOf(i);
					const strike = k >= 0 ? interpolate(f - markStart - k * (STRIKE - 2), [0, STRIKE], [0, 1], clamp) : 0;
					return (
						<span key={i} style={{position: 'relative', display: 'inline-block'}}>
							{visible}
							{k >= 0 ? (
								<svg style={{position: 'absolute', left: -6, top: 0, overflow: 'visible'}} width={t.w.length * size * 0.6 + 12} height={size * 1.2}>
									<path
										d={`M2 ${size * 0.78} L${t.w.length * size * 0.6 + 10} ${size * 0.38}`}
										stroke={E.red}
										strokeWidth={3.2}
										strokeLinecap="round"
										fill="none"
										pathLength={1}
										strokeDasharray={1}
										strokeDashoffset={1 - strike}
									/>
								</svg>
							) : null}
						</span>
					);
				})}
			</div>
			{bad.map((_, k) => (
				<Sequence key={k} from={markStart + k * (STRIKE - 2)} durationInFrames={12} layout="none">
					<Audio src={staticFile('sfx/pen.wav')} volume={0.45} />
				</Sequence>
			))}
			<ExaminerScore at={scoreAt} text={`${right}/${refWords}`} top={size * 0.5} />
		</div>
	);
};

// The examiner's handwritten score, ringed, in the right-hand column.
export const ExaminerScore: React.FC<{at: number; text: string; top: number}> = ({at, text, top}) => {
	const f = useCurrentFrame();
	const write = interpolate(f - at, [0, 10], [0, 1], clamp);
	const ring = interpolate(f - at - 8, [0, 12], [0, 1], clamp);
	return (
		<div style={{position: 'absolute', left: COL - LEFT + 30, top, width: 140, height: 70, opacity: write > 0 ? 1 : 0}}>
			<div style={{position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: HAND, fontWeight: 700, fontSize: 40, color: E.red, clipPath: `inset(0 ${(1 - write) * 100}% 0 0)`}}>
				{text}
			</div>
			<svg style={{position: 'absolute', inset: 0, overflow: 'visible'}} width={140} height={70}>
				<path d="M120 22 C 116 4, 30 2, 14 22 C 2 40, 26 66, 74 64 C 118 62, 134 44, 112 18" fill="none" stroke={E.red} strokeWidth={2.6} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - ring} />
			</svg>
		</div>
	);
};

// A tick drawn by the examiner.
export const Tick: React.FC<{at: number; size?: number}> = ({at, size = 34}) => {
	const f = useCurrentFrame();
	const p = interpolate(f - at, [0, 8], [0, 1], clamp);
	return (
		<svg width={size} height={size} viewBox="0 0 34 34" style={{overflow: 'visible'}}>
			<path d="M3 18 L12 28 L31 4" fill="none" stroke={E.red} strokeWidth={3.4} strokeLinecap="round" strokeLinejoin="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - p} />
		</svg>
	);
};

// Printed box with a heading rule, as used for mark schemes and sources.
export const Box: React.FC<{title: string; children: React.ReactNode; style?: React.CSSProperties; at?: number}> = ({title, children, style, at = 0}) => {
	const f = useCurrentFrame();
	const o = interpolate(f - at, [0, 10], [0, 1], clamp);
	return (
		<div style={{position: 'absolute', border: `2px solid ${E.rule}`, opacity: o, ...style}}>
			<div style={{padding: '12px 22px', borderBottom: `1.5px solid ${E.rule}`, fontSize: 26, fontWeight: 700}}>{title}</div>
			<div style={{padding: '18px 22px'}}>{children}</div>
		</div>
	);
};

export const Fade: React.FC<{at: number; children: React.ReactNode; style?: React.CSSProperties; dur?: number}> = ({at, children, style, dur = 12}) => {
	const f = useCurrentFrame();
	const t = easeOut((f - at) / dur);
	return <div style={{opacity: t, transform: `translateY(${(1 - t) * 10}px)`, ...style}}>{children}</div>;
};
