import {loadFont} from '@remotion/google-fonts/SchibstedGrotesk';
import {AbsoluteFill, interpolate, useCurrentFrame} from 'remotion';
import type {Clip, Token} from '../../data';
import {BIGGER, data, severityWords, STOCK, wordCer} from '../../data';
import {easeOut, pct, useWave} from '../../shared/audio';
import type {SceneProps} from '../../shared/Film';
import {Film, Watermark} from '../../shared/Film';
import {clipEnd, OVERLAP} from '../../shared/timeline';

const {fontFamily: G} = loadFont('normal', {weights: ['400', '500', '800'], subsets: ['latin']});

const P = {ground: '#f3f3f1', ink: '#0e0e0e', mid: '#6f6f6b', rule: '#cfcfca', faint: '#e4e4e0', accent: '#2f5bea', wrong: '#9c9c97'};
const M = 140; // page margin
const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const SYSTEMS = [
	{key: STOCK, label: `${STOCK} (stock)`, color: P.ink},
	{key: 'whisper-large-v3', label: 'whisper-large-v3 (stock)', color: P.ink},
	{key: 'Clear Voice', label: 'Clear Voice', color: P.accent},
];

// Each scene fades in over the last; the claim lands first, the proof follows.
const Frame: React.FC<{children: React.ReactNode; first?: boolean}> = ({children, first}) => {
	const f = useCurrentFrame();
	const o = first ? 1 : easeOut(f / OVERLAP);
	return <AbsoluteFill style={{backgroundColor: P.ground, opacity: o, fontFamily: G, color: P.ink}}>{children}</AbsoluteFill>;
};

const Claim: React.FC<{children: React.ReactNode; at?: number; size?: number; top?: number}> = ({children, at = 4, size = 80, top = 120}) => {
	const f = useCurrentFrame();
	const t = easeOut((f - at) / 16);
	return (
		<div style={{position: 'absolute', top, left: M, maxWidth: 1500, fontSize: size, fontWeight: 800, lineHeight: 1.08, letterSpacing: '-0.03em', ...({textWrap: 'balance'} as React.CSSProperties), opacity: t, transform: `translateY(${(1 - t) * 18}px)`}}>
			{children}
		</div>
	);
};

const Proof: React.FC<{at: number; children: React.ReactNode; style?: React.CSSProperties}> = ({at, children, style}) => {
	const f = useCurrentFrame();
	const t = easeOut((f - at) / 18);
	return <div style={{position: 'absolute', opacity: t, transform: `translateY(${(1 - t) * 24}px)`, ...style}}>{children}</div>;
};

const Line: React.FC<{label: string; color: string; tokens: Token[]; at: number; size?: number}> = ({label, color, tokens, at, size = 46}) => {
	const f = useCurrentFrame();
	return (
		<div style={{display: 'flex', alignItems: 'baseline', borderTop: `1.5px solid ${P.rule}`, padding: '20px 0'}}>
			<div style={{width: 380, fontSize: 26, fontWeight: 500, color: color === P.accent ? P.accent : P.mid}}>{label}</div>
			<div style={{flex: 1, display: 'flex', flexWrap: 'wrap', gap: `0 ${size * 0.3}px`, fontSize: size, fontWeight: 500, letterSpacing: '-0.01em'}}>
				{tokens.length === 0 ? <span style={{color: P.wrong}}>nothing</span> : null}
				{tokens.map((t, i) => {
					const o = interpolate(f - at - i * 2, [0, 8], [0, 1], clamp);
					const bad = t.tag !== 'ok';
					return (
						<span key={i} style={{opacity: o, color: bad ? P.wrong : color, textDecoration: bad ? `line-through ${P.ink} 2px` : 'none'}}>
							{t.w}
						</span>
					);
				})}
			</div>
		</div>
	);
};

const Wave: React.FC<{clip: Clip; playAt: number; width: number}> = ({clip, playAt, width}) => {
	const {bars, progress} = useWave(clip.file, clip.duration, playAt, 240);
	const h = 90;
	const d = bars.map((a, i) => `${i ? 'L' : 'M'}${(i / Math.max(1, bars.length - 1)) * width} ${h / 2 - (a * h) / 2}`).join(' ');
	const d2 = bars.map((a, i) => `${i ? 'L' : 'M'}${(i / Math.max(1, bars.length - 1)) * width} ${h / 2 + (a * h) / 2}`).join(' ');
	return (
		<svg width={width} height={h} style={{overflow: 'visible'}}>
			<defs>
				<clipPath id={`played-${clip.rowId}`}>
					<rect width={progress * width} height={h} />
				</clipPath>
			</defs>
			<g stroke={P.rule} strokeWidth={2} fill="none">
				<path d={d} />
				<path d={d2} />
			</g>
			<g stroke={P.ink} strokeWidth={2} fill="none" clipPath={`url(#played-${clip.rowId})`}>
				<path d={d} />
				<path d={d2} />
			</g>
		</svg>
	);
};

const Transcripts: React.FC<{clip: Clip; at: number; top: number; systems?: string[]}> = ({clip, at, top, systems = SYSTEMS.map((s) => s.key)}) => (
	<Proof at={at} style={{top, left: M, right: M}}>
		{SYSTEMS.filter((s) => systems.includes(s.key) && clip.systems[s.key]).map((s, i) => (
			<Line key={s.key} label={s.label} color={s.color} tokens={clip.systems[s.key].tokens} at={at + 8 + i * 22} />
		))}
		<Line label="What was said" color={P.ink} tokens={clip.ref.split(' ').map((w) => ({w, tag: 'ok' as const}))} at={at + 8 + systems.length * 22} />
	</Proof>
);

const Bars: React.FC<{rows: {label: string; v: number; color: string}[]; at: number; top: number; width?: number}> = ({rows, at, top, width = 1100}) => {
	const f = useCurrentFrame();
	const max = Math.max(...rows.map((r) => r.v)) * 1.08;
	return (
		<div style={{position: 'absolute', top, left: M, right: M}}>
			{rows.map((r, i) => {
				const t = easeOut((f - at - i * 12) / 26);
				return (
					<div key={r.label} style={{display: 'flex', alignItems: 'center', borderTop: `1.5px solid ${P.rule}`, padding: '24px 0'}}>
						<div style={{width: 420, fontSize: 30, fontWeight: 500, color: r.color === P.accent ? P.accent : P.ink}}>{r.label}</div>
						<div style={{width: width, height: 34, position: 'relative'}}>
							<div style={{position: 'absolute', left: 0, top: 0, height: 34, width: (r.v / max) * width * t, background: r.color}} />
						</div>
						<div style={{marginLeft: 28, fontSize: 44, fontWeight: 800, fontVariantNumeric: 'tabular-nums', opacity: t}}>{pct(r.v)}</div>
					</div>
				);
			})}
		</div>
	);
};

const Title: React.FC<SceneProps> = () => (
	<Frame first>
		<Claim at={6} size={44} top={120}>
			Clear Voice
		</Claim>
		<Claim at={14} size={112} top={300}>
			Speech recognition fine-tuned for people with dysarthria.
		</Claim>
		<Proof at={40} style={{bottom: 120, left: M, fontSize: 26, color: P.mid}}>
			Tested on voices and sentences it never heard. Veritas AI Hackathon, Languages track.
		</Proof>
		<Watermark color={P.accent} font={G} />
	</Frame>
);

// For a one-word clip: how the two stock models did, in words.
const stockVerdict = (clip: Clip): string => {
	const wrong = [STOCK, 'whisper-large-v3'].filter((k) => clip.systems[k] && clip.systems[k].wer > 0).length;
	return wrong === 2 ? 'Both stock Whisper models got the word wrong.' : wrong === 1 ? 'One of the two stock Whisper models got the word wrong.' : 'Both stock Whisper models got the word right.';
};

const Hook: React.FC<SceneProps> = ({part}) => {
	const clip = part.clip!;
	const end = clipEnd(part);
	const stock = clip.systems[STOCK];
	const n = clip.ref.split(' ').length;
	const right = stock ? stock.tokens.filter((t) => t.tag === 'ok').length : 0;
	return (
		<Frame>
			<Claim at={4} size={64}>
				A speaker with {severityWords(clip.severity)} dysarthria. The model has never heard this speaker or this {n === 1 ? 'word' : 'sentence'}.
			</Claim>
			<Proof at={20} style={{top: 340, left: M}}>
				<Wave clip={clip} playAt={part.playAt!} width={1640} />
			</Proof>
			<Claim at={end + 8} size={44} top={470}>
				{n === 1 ? stockVerdict(clip) : `Stock ${STOCK} got ${right} of ${n} words right.`}
			</Claim>
			<Transcripts clip={clip} at={end + 20} top={560} systems={[STOCK, 'whisper-large-v3', 'Clear Voice']} />
		</Frame>
	);
};

const Problem: React.FC<SceneProps> = () => {
	const dys = data.headline.find((b) => b.name === STOCK)!.value;
	const typ = data.stockTypical;
	return (
		<Frame>
			<Claim size={88}>Dictation was built on typical voices.</Claim>
			<Proof at={24} style={{top: 360, left: M, width: 1300, fontSize: 34, lineHeight: 1.45, color: P.mid}}>
				Dysarthria makes speech slow, slurred or quiet. It can come with cerebral palsy, ALS, a stroke or Parkinson’s. Stock {STOCK}, typical voices against voices with dysarthria:
			</Proof>
			<Bars
				at={60}
				top={560}
				rows={[...(typ !== undefined ? [{label: 'Typical voices', v: typ, color: P.mid}] : []), {label: 'Voices with dysarthria', v: dys, color: P.ink}]}
			/>
			<Proof at={110} style={{bottom: 110, left: M, fontSize: 22, color: P.mid}}>
				Share of words wrong. TORGO database, speakers and sentences not used in training.
			</Proof>
		</Frame>
	);
};

const Demo: React.FC<SceneProps> = ({part}) => {
	const clip = part.clip!;
	const end = clipEnd(part);
	return (
		<Frame>
			<Claim size={64}>
				{severityWords(clip.severity)[0].toUpperCase() + severityWords(clip.severity).slice(1)} dysarthria, picked at random.
			</Claim>
			<Proof at={12} style={{top: 250, left: M}}>
				<Wave clip={clip} playAt={part.playAt!} width={1640} />
			</Proof>
			<Transcripts clip={clip} at={end + 8} top={400} />
		</Frame>
	);
};

const How: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	const rows = 15;
	const cols = 20;
	const cw = 52;
	const ch = 30;
	const x = M;
	const y = 380;
	const group = (r: number) => (r < 4 ? 'A' : r < 8 ? 'B' : 'typical');
	return (
		<Frame>
			<Claim size={72}>Tested only on people and sentences it never heard.</Claim>
			<svg style={{position: 'absolute', left: 0, top: 0}} width={1920} height={1080}>
				{Array.from({length: rows}, (_, r) =>
					Array.from({length: cols}, (_, c) => {
						const test = group(r) === 'A' && c >= 16;
						const train = group(r) !== 'A' && c < 14;
						const appear = easeOut((f - 30 - (r + c) * 1.1) / 10);
						const testIn = test ? easeOut((f - 120) / 16) : 1;
						return (
							<rect
								key={`${r}-${c}`}
								x={x + c * cw}
								y={y + r * ch}
								width={cw - 4}
								height={ch - 4}
								fill={test ? P.accent : train ? P.ink : P.faint}
								opacity={appear * (test ? testIn : train ? 0.85 : 1)}
							/>
						);
					}),
				)}
			</svg>
			<Proof at={60} style={{top: y, left: x + cols * cw + 60, width: 560, fontSize: 30, lineHeight: 1.45}}>
				<div>Rows are speakers. Columns are groups of sentences.</div>
				<div style={{marginTop: 22}}>
					<b>Black:</b> training.
				</div>
				<div style={{color: P.accent}}>
					<b>Blue:</b> the test.
				</div>
				<div style={{color: P.mid}}>
					<b>Grey:</b> not used, or the 10% of sentences kept for choosing when to stop training.
				</div>
				<div style={{marginTop: 22, color: P.mid}}>No overlap in either direction. Then the groups swap.</div>
			</Proof>
		</Frame>
	);
};

const Results: React.FC<SceneProps> = () => {
	const bars = data.headline;
	const ours = bars.find((b) => b.name === 'Clear Voice')!.value;
	const stock = bars.find((b) => b.name === STOCK)!.value;
	const large = bars.find((b) => b.name === 'whisper-large-v3')?.value ?? Infinity;
	const claim =
		ours < Math.min(stock, large)
			? `${Math.round((1 - ours / stock) * 100)}% fewer words wrong than stock Whisper.`
			: ours < stock
				? 'Fewer errors than the model it started from. Not as few as the largest.'
				: 'Not yet better than stock Whisper.';
	return (
		<Frame>
			<Claim size={80}>{claim}</Claim>
			<Bars
				at={30}
				top={420}
				rows={bars.map((b) => ({label: SYSTEMS.find((s) => s.key === b.name)?.label ?? b.name, v: b.value, color: b.name === 'Clear Voice' ? P.accent : P.ink}))}
			/>
			<Proof at={90} style={{bottom: 110, left: M, fontSize: 22, color: P.mid}}>
				Share of words wrong on sentences, new speakers with dysarthria saying new sentences. With 8 test speakers, the 95% intervals overlap (see the write-up).
			</Proof>
			{wordCer('Clear Voice') !== undefined && wordCer(STOCK) !== undefined ? (
				<Proof at={70} style={{bottom: 170, left: M, fontSize: 34, fontWeight: 500}}>
					Single words: {pct(wordCer('Clear Voice')!)} of letters wrong, against {pct(wordCer(STOCK)!)} for stock Whisper.
				</Proof>
			) : null}
		</Frame>
	);
};

const Honest: React.FC<SceneProps> = () => {
	const seen = data.leakage.find((b) => /seen sentence/i.test(b.name))!.value;
	const strict = data.leakage.find((b) => /new speaker, new/i.test(b.name))!.value;
	return (
		<Frame>
			<Claim size={80}>{strict > seen ? `Practised sentences would flatter it by ${Math.round((strict - seen) * 100)} points.` : 'Practised sentences gave it no advantage here.'}</Claim>
			<Bars
				at={30}
				top={420}
				rows={[
					{label: 'Sentences it practised on', v: seen, color: P.mid},
					{label: 'Sentences it never saw', v: strict, color: P.accent},
				]}
			/>
			<Proof at={80} style={{top: 760, left: M, fontSize: 34, fontWeight: 500}}>
				The number we report is the second one.
			</Proof>
		</Frame>
	);
};

const Personal: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	const s = data.personalisation[0];
	if (!s) return <Frame>{null}</Frame>;
	const first = s.points[0];
	const last = s.points[s.points.length - 1];
	const W = 1200;
	const H = 420;
	const x0 = M + 60;
	const y0 = 430;
	const maxY = Math.max(...s.points.map((p) => p.wer)) * 1.15;
	const X = (m: number) => x0 + (m / 10) * W;
	const Y = (v: number) => y0 + (1 - v / maxY) * H;
	const draw = easeOut((f - 40) / 50);
	return (
		<Frame>
			<Claim size={76}>
				{({2: 'Two', 5: 'Five', 10: 'Ten'} as Record<number, string>)[last.minutes] ?? last.minutes} minutes of one person’s voice: {pct(first.wer)} to {pct(last.wer)} words wrong.
			</Claim>
			<svg style={{position: 'absolute', inset: 0}} width={1920} height={1080}>
				<line x1={x0} x2={x0 + W} y1={y0 + H} y2={y0 + H} stroke={P.rule} strokeWidth={2} />
				<path d={s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p.minutes)},${Y(p.wer)}`).join(' ')} fill="none" stroke={P.accent} strokeWidth={5} pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
				{s.points.map((p, i) =>
					draw >= i / (s.points.length - 1) - 0.01 ? (
						<g key={i}>
							<circle cx={X(p.minutes)} cy={Y(p.wer)} r={10} fill={P.accent} />
							<text x={X(p.minutes)} y={Y(p.wer) - 26} textAnchor="middle" fontFamily={G} fontSize={30} fontWeight={800} fill={P.ink}>
								{pct(p.wer)}
							</text>
							<text x={X(p.minutes)} y={y0 + H + 44} textAnchor="middle" fontFamily={G} fontSize={26} fill={P.mid}>
								{p.minutes === 0 ? 'none' : `${p.minutes} min`}
							</text>
						</g>
					) : null,
				)}
			</svg>
			<Proof at={120} style={{top: 470, left: x0 + W + 80, width: 360, fontSize: 28, lineHeight: 1.45, color: P.mid}}>
				Speaker {s.speaker}, {s.severity} dysarthria, tested on sentences that were not in the added recordings.
			</Proof>
			{data.personalisation.slice(1).map((o) => (
				<Proof key={o.speaker} at={150} style={{bottom: 110, left: M, fontSize: 34, fontWeight: 500}}>
					Speaker {o.speaker}, {o.severity}: {pct(o.points[0].wer)} before, {pct(o.points[o.points.length - 1].wer)} after {o.points[o.points.length - 1].minutes} minutes.
				</Proof>
			))}
		</Frame>
	);
};

const Close: React.FC<SceneProps> = () => (
	<Frame>
		<Claim size={112} top={200}>
			Clear Voice
		</Claim>
		<Proof at={16} style={{top: 360, left: M, width: 1400, fontSize: 40, lineHeight: 1.4}}>
			Next: try it with speech therapists, on real dictation.
		</Proof>
		<Proof at={34} style={{top: 520, left: M, fontSize: 32, lineHeight: 1.6}}>
			<div style={{fontWeight: 800}}>{data.team}</div>
			<div>{data.repoUrl}</div>
			{data.demoUrl ? <div>{data.demoUrl}</div> : null}
		</Proof>
		<Proof at={56} style={{bottom: 100, left: M, right: M, fontSize: 21, color: P.mid, lineHeight: 1.5}}>
			Recordings from the TORGO database (Rudzicz, Namasivayam and Wolff, 2012), non-commercial research use. Models: OpenAI Whisper. Research prototype, not a medical device. Built with help from Claude (Anthropic).
		</Proof>
	</Frame>
);

export const ProofFilm: React.FC = () => (
	<Film
		ground={P.ground}
		scenes={{title: Title, hook: Hook, problem: Problem, demo: Demo, how: How, results: Results, honest: Honest, personal: Personal, close: Close}}
		music={{file: 'sfx/music.wav', volume: 0.24}}
		turn={{file: 'sfx/whoosh.wav', volume: 0.12, offset: -6}}
	/>
);
