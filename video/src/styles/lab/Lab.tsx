import {Img, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {BIGGER, BIGGER_WORDS, data, severityWords, STOCK, wordCer} from '../../data';
import {easeOut, pct} from '../../shared/audio';
import type {SceneProps} from '../../shared/Film';
import {Film, Watermark} from '../../shared/Film';
import {Epilogue, PaperFigures, Prologue} from '../../shared/Human';
import {clipEnd} from '../../shared/timeline';
import {Analysis, Caption, L, MONO, Rise, SANS, Sheet, SYSTEMS} from './parts';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;

// A figure heading with its one-line description underneath, kept together however the heading wraps.
// Honest one-line summary of the personalisation curves, e.g. "Ten minutes of a speaker’s own voice: a small gain for M01, none for F03".
const personalHeadline = (series: {speaker: string; points: {minutes: number; wer: number}[]}[]): string => {
	if (!series.length) return 'A speaker’s own voice';
	const mins = series[0].points[series[0].points.length - 1].minutes;
	const word = ({2: 'Two', 5: 'Five', 10: 'Ten'} as Record<number, string>)[mins] ?? String(mins);
	const gain = (s: {speaker: string; points: {wer: number}[]}) => {
		const d = Math.round((s.points[0].wer - s.points[s.points.length - 1].wer) * 100);
		return d >= 10 ? `a clear gain for ${s.speaker}` : d >= 3 ? `a small gain for ${s.speaker}` : `none for ${s.speaker}`;
	};
	return `${word} minutes of a speaker’s own voice: ${series.map(gain).join(', ')}`;
};

const Head: React.FC<{children: React.ReactNode; sub: string; at: number}> = ({children, sub, at}) => (
	<div style={{position: 'absolute', top: 70, left: 100, right: 100, display: 'flex', flexDirection: 'column', gap: 14}}>
		<Rise at={at} style={{fontSize: 50, fontWeight: 800, lineHeight: 1.12, maxWidth: 1600, ...({textWrap: 'balance'} as React.CSSProperties)}}>
			{children}
		</Rise>
		<Rise at={at + 10} style={{fontSize: 28, color: L.muted}}>
			{sub}
		</Rise>
	</div>
);
const colorOf = (name: string) => SYSTEMS.find((s) => s.key === name)?.color ?? L.ink;

// A horizontal axis for error rates, with gridlines at every 10% (or 20% when the range is wide).
const ErrAxis: React.FC<{x: number; y: number; w: number; h: number; max: number; at: number}> = ({x, y, w, h, max, at}) => {
	const f = useCurrentFrame();
	const step = max > 0.6 ? 0.2 : 0.1;
	const ticks = Array.from({length: Math.floor(max / step + 1e-9) + 1}, (_, i) => i * step);
	const o = interpolate(f - at, [0, 10], [0, 1], clamp);
	return (
		<svg style={{position: 'absolute', left: x, top: y, overflow: 'visible', opacity: o}} width={w} height={h}>
			{ticks.map((t) => (
				<g key={t}>
					<line x1={(t / max) * w} x2={(t / max) * w} y1={0} y2={h} stroke={L.grid} strokeWidth={1.5} />
					<text x={(t / max) * w} y={h + 34} textAnchor="middle" fontFamily={MONO} fontSize={22} fill={L.muted}>
						{Math.round(t * 100)}%
					</text>
				</g>
			))}
		</svg>
	);
};

const Title: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	const sweep = interpolate(f, [20, 160], [0, 1], clamp);
	return (
		<Sheet first>
			<div style={{position: 'absolute', top: 150, left: 100}}>
				<Rise at={4} style={{fontSize: 150, fontWeight: 800, letterSpacing: '-0.025em', lineHeight: 1}}>
					Clear Voice
				</Rise>
				<Rise at={16} style={{fontSize: 40, marginTop: 24, maxWidth: 1300, lineHeight: 1.35}}>
					Speech recognition for people with dysarthria, tested on voices and sentences it never heard.
				</Rise>
			</div>
			<div style={{position: 'absolute', top: 560, left: 100, right: 100, height: 300, overflow: 'hidden', opacity: 0.9}}>
				{data.opening.spec ? <Img src={staticFile(data.opening.spec)} style={{width: '100%', height: 300}} /> : null}
				<div style={{position: 'absolute', top: 0, bottom: 0, left: `${sweep * 100}%`, width: 2, background: L.cursor}} />
			</div>
			<Rise at={30} style={{position: 'absolute', top: 880, left: 100, fontSize: 24, color: L.muted}}>
				Spectrogram of a recording from the test set. Veritas AI Hackathon, Languages track.
			</Rise>
			<Watermark color={L.cursor} font={SANS} />
		</Sheet>
	);
};

const Hook: React.FC<SceneProps> = ({part}) => {
	const clip = part.clip!;
	return (
		<Sheet>
			<Analysis clip={clip} playAt={part.playAt!} heading={`What did they say? Speaker ${clip.speaker}, ${severityWords(clip.severity)} dysarthria`} tiersAt={clipEnd(part) + 12} />
			<Caption n={1} at={clipEnd(part) + 90}>
				A recording the models never heard in training. Misheard words are shaded; numbers on the right are word error rates for this clip.
			</Caption>
		</Sheet>
	);
};

const Problem: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	const dys = data.headline.find((b) => b.name === STOCK)!.value;
	const typ = data.stockTypical;
	const rows = [...(typ !== undefined ? [{label: 'Typical voices', v: typ}] : []), {label: 'Voices with dysarthria', v: dys}];
	const max = Math.max(...rows.map((r) => r.v)) * 1.15;
	const W = 760;
	return (
		<Sheet>
			<Rise at={4} style={{position: 'absolute', top: 120, left: 100, width: 720, fontSize: 48, fontWeight: 800, lineHeight: 1.2}}>
				Speech recognition was built on typical voices.
			</Rise>
			<Rise at={20} style={{position: 'absolute', top: 290, left: 100, width: 700, fontSize: 32, lineHeight: 1.5}}>
				Dysarthria makes speech slow, slurred or quiet. It can come with cerebral palsy, ALS, a stroke or Parkinson’s disease. Dictation and live captions often turn it into the wrong words.
			</Rise>
			<div style={{position: 'absolute', top: 300, left: 960, width: W + 200}}>
				<div style={{fontSize: 26, fontWeight: 700, marginBottom: 30}}>Stock {STOCK}, words wrong</div>
				<div style={{position: 'relative', height: rows.length * 120}}>
					<ErrAxis x={260} y={0} w={W - 260} h={rows.length * 120} max={max} at={30} />
					{rows.map((r, i) => {
						const t = easeOut((f - 50 - i * 30) / 30);
						return (
							<div key={r.label} style={{position: 'absolute', top: i * 120 + 30, left: 0, display: 'flex', alignItems: 'center'}}>
								<div style={{width: 250, fontSize: 26}}>{r.label}</div>
								<div style={{width: ((r.v / max) * (W - 260)) * t, height: 40, background: i === rows.length - 1 ? L.ink : L.small}} />
								<div style={{marginLeft: 16, fontFamily: MONO, fontSize: 30, fontWeight: 600, opacity: t}}>{pct(r.v)}</div>
							</div>
						);
					})}
				</div>
			</div>
			<Caption n={2} at={120}>
				Stock {STOCK} on the TORGO database, sentences only, speakers it was not trained on. Word error rate: the share of words it gets wrong.
			</Caption>
		</Sheet>
	);
};

const Demo: React.FC<SceneProps> = ({part}) => {
	const clip = part.clip!;
	const n = 3 + (part.index ?? 0);
	return (
		<Sheet>
			<Analysis clip={clip} playAt={part.playAt!} heading={`Speaker ${clip.speaker}, ${severityWords(clip.severity)} dysarthria, chosen at random`} tiersAt={clipEnd(part) + 10} />
			<Caption n={n} at={clipEnd(part) + 70}>
				Random clip from the strict test set: a speaker and a sentence that never appeared in training.
			</Caption>
		</Sheet>
	);
};

// Speakers (rows) by sentence groups (columns). Training uses the grey cells; the test is only the blue block.
const How: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const rows = [
		...['F01', 'M01', 'F03', 'M03'].map((s) => ({s, group: 'A'})),
		...['M02', 'M04', 'M05', 'F04'].map((s) => ({s, group: 'B'})),
		...['FC01', 'MC01', 'MC02', 'FC02', 'FC03', 'MC03', 'MC04'].map((s) => ({s, group: 'typical'})),
	];
	const cols = 20; // 14 train, 2 validation, 4 test sentence groups
	const role = (r: number, c: number, held: 'A' | 'B') => {
		const g = rows[r].group;
		const test = c >= 16;
		const val = c >= 14 && c < 16;
		if (g === held) return test ? 'test' : 'none';
		return test ? 'none' : val ? 'val' : 'train';
	};
	const swap = f > 260;
	const held = swap ? 'B' : 'A';
	const cw = 46;
	const ch = 30;
	const gx = 360;
	const gy = 230;
	const fill = (r: string) => (r === 'test' ? L.clear : r === 'train' ? '#b9b9b9' : r === 'val' ? '#e2e2e2' : '#ffffff');
	const n = 3 + (part.total ?? 1);
	return (
		<Sheet>
			<Rise at={4} style={{position: 'absolute', top: 70, left: 100, fontSize: 48, fontWeight: 800}}>
				How the test is kept honest
			</Rise>
			<Rise at={14} style={{position: 'absolute', top: 140, left: 100, fontSize: 28, color: L.muted}}>
				Each row is a speaker; each column is a group of sentences. {data.nRecordings.toLocaleString('en-US')} recordings in all.
			</Rise>
			<svg style={{position: 'absolute', left: 0, top: 0}} width={1920} height={1080}>
				{rows.map((row, r) => (
					<text key={row.s} x={gx - 20} y={gy + r * ch + ch * 0.72} textAnchor="end" fontFamily={MONO} fontSize={20} fill={row.group === 'typical' ? L.muted : L.ink}>
						{row.s}
					</text>
				))}
				{rows.map((_, r) =>
					Array.from({length: cols}, (_, c) => {
						const rl = role(r, c, held);
						const appear = easeOut((f - 30 - (r + c) * 1.2) / 12);
						const testIn = rl === 'test' ? easeOut((f - (swap ? 270 : 130) - c * 2) / 12) : 1;
						const color = rl === 'test' ? `rgba(35,87,198,${testIn})` : fill(rl);
						return <rect key={`${r}-${c}`} x={gx + c * cw} y={gy + r * ch} width={cw - 3} height={ch - 3} fill={color} stroke={L.grid} strokeWidth={1} opacity={appear} />;
					}),
				)}
				<line x1={gx} x2={gx + cols * cw} y1={gy + 4 * ch - 1.5} y2={gy + 4 * ch - 1.5} stroke={L.ink} strokeWidth={2} />
				<line x1={gx} x2={gx + cols * cw} y1={gy + 8 * ch - 1.5} y2={gy + 8 * ch - 1.5} stroke={L.ink} strokeWidth={2} />
				<text x={gx + 7 * cw} y={gy - 16} textAnchor="middle" fontFamily={SANS} fontSize={22} fill={L.muted}>
					70% of sentences
				</text>
				<text x={gx + 15 * cw} y={gy - 16} textAnchor="middle" fontFamily={SANS} fontSize={22} fill={L.muted}>
					10%
				</text>
				<text x={gx + 18 * cw} y={gy - 16} textAnchor="middle" fontFamily={SANS} fontSize={22} fill={L.muted}>
					20%
				</text>
			</svg>
			<div style={{position: 'absolute', left: gx + cols * cw + 60, top: gy, width: 520, fontSize: 28, lineHeight: 1.45}}>
				{[
					['#b9b9b9', 'Training'],
					['#e2e2e2', 'Choosing the checkpoint'],
					[L.clear, 'Test: new speakers, new sentences'],
					['#ffffff', 'Never used in this round'],
				].map(([c, t], i) => (
					<Rise key={t} at={60 + i * 12} style={{display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14}}>
						<div style={{width: 30, height: 22, background: c, border: `1px solid ${L.grid}`}} />
						{t}
					</Rise>
				))}
				<Rise at={170} style={{marginTop: 30, fontWeight: 700}}>
					No test speaker and no test sentence is ever trained on.
				</Rise>
				<Rise at={262} style={{marginTop: 18, color: L.muted}}>
					Then the two groups swap, so every speaker is tested once.
				</Rise>
			</div>
			<Caption n={n} at={200}>
				Near-identical prompts (“as I choose”, “as I chose”) are kept in the same column group, so a test sentence cannot be recalled from training.
			</Caption>
		</Sheet>
	);
};

// Dot and interval for each system on the strict test.
const Results: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const bars = data.headline;
	const max = Math.ceil((Math.max(...bars.map((b) => b.hi ?? b.value)) * 1.1) / 0.1) * 0.1;
	const ours = bars.find((b) => b.name === 'Clear Voice')!.value;
	const stock = bars.find((b) => b.name === STOCK)!.value;
	const large = bars.find((b) => b.name === 'whisper-large-v3')?.value ?? Infinity;
	const heading =
		ours < Math.min(stock, large)
			? 'Fewer words wrong than either stock model'
			: ours < stock
				? `Fewer errors than its base model, though the ranges overlap; the ${BIGGER} larger model still leads`
				: 'Not yet better than stock Whisper';
	const x = 560;
	const w = 1100;
	const rowH = 130;
	const n = 4 + (part.total ?? 1);
	return (
		<Sheet>
			<Head at={4} sub="Speakers with dysarthria the models never heard, saying sentences they never heard.">
				{heading}
			</Head>
			<ErrAxis x={x} y={280} w={w} h={rowH * bars.length} max={max} at={20} />
			{bars.map((b, i) => {
				const t = easeOut((f - 40 - i * 20) / 24);
				const cy = 280 + i * rowH + rowH / 2;
				const color = colorOf(b.name);
				return (
					<div key={b.name}>
						<div style={{position: 'absolute', top: cy - 22, left: 100, width: 440, fontSize: 32, fontWeight: b.name === 'Clear Voice' ? 800 : 600, color}}>{b.name}</div>
						<svg style={{position: 'absolute', left: x, top: cy - 20, overflow: 'visible'}} width={w} height={40}>
							{b.lo !== undefined && b.hi !== undefined ? (
								<line x1={(b.lo / max) * w} x2={(b.lo / max) * w + ((b.hi - b.lo) / max) * w * t} y1={20} y2={20} stroke={color} strokeWidth={4} />
							) : null}
							<circle cx={(b.value / max) * w} cy={20} r={14 * t} fill={color} />
						</svg>
						<div style={{position: 'absolute', top: cy - 22, left: x + w + 30, fontFamily: MONO, fontSize: 30, fontWeight: 600, opacity: t}}>{pct(b.value)}</div>
					</div>
				);
			})}
			<Rise at={20} style={{position: 'absolute', top: 280 + rowH * bars.length + 50, left: x, width: w, textAlign: 'center', fontSize: 24, color: L.muted}}>
				Word error rate on sentences
			</Rise>
			{wordCer('Clear Voice') !== undefined && wordCer(STOCK) !== undefined ? (
				<Rise at={70} style={{position: 'absolute', top: 280 + rowH * bars.length + 100, left: 100, right: 100, fontSize: 32}}>
					Single words: Clear Voice gets <b style={{color: L.clear}}>{pct(wordCer('Clear Voice')!)}</b> of letters wrong; stock {STOCK} {pct(wordCer(STOCK)!)}
					{wordCer('whisper-large-v3') !== undefined ? `, whisper-large-v3 ${pct(wordCer('whisper-large-v3')!)}` : ''}.
				</Rise>
			) : null}
			<Caption n={n} at={100}>
				Dots are word error rates; lines are 95% intervals from resampling speakers. whisper-large-v3 is a stock model about {BIGGER_WORDS} the size of {STOCK}.
			</Caption>
		</Sheet>
	);
};

const Honest: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const seen = data.leakage.find((b) => /seen sentence/i.test(b.name))!.value;
	const strict = data.leakage.find((b) => /new speaker, new/i.test(b.name))!.value;
	const max = Math.ceil((strict * 1.4) / 0.1) * 0.1;
	const x = 360;
	const w = 1200;
	const a = easeOut((f - 40) / 24);
	const b = easeOut((f - 80) / 24);
	const n = 5 + (part.total ?? 1);
	return (
		<Sheet>
			<Head at={4} sub="Clear Voice, the same new speakers. Only the sentences change.">
				{strict > seen ? `Testing on practised sentences would have flattered it by ${Math.round((strict - seen) * 100)} points` : 'Practised sentences gave it no advantage here'}
			</Head>
			<ErrAxis x={x} y={330} w={w} h={260} max={max} at={20} />
			<svg style={{position: 'absolute', left: x, top: 330, overflow: 'visible'}} width={w} height={260}>
				<line x1={(seen / max) * w} x2={(seen / max) * w + ((strict - seen) / max) * w * b} y1={130} y2={130} stroke={L.ink} strokeWidth={3} strokeDasharray="8 8" />
				<circle cx={(seen / max) * w} cy={130} r={16 * a} fill="#ffffff" stroke={L.clear} strokeWidth={5} />
				<circle cx={(strict / max) * w} cy={130} r={16 * b} fill={L.clear} />
			</svg>
			<Rise at={50} style={{position: 'absolute', left: x + (seen / max) * w - 210, top: 520, width: 420, textAlign: 'center', fontSize: 28}}>
				Sentences it practised on <b style={{fontFamily: MONO}}>{pct(seen)}</b>
			</Rise>
			<Rise at={90} style={{position: 'absolute', left: x + (strict / max) * w - 210, top: 380, width: 420, textAlign: 'center', fontSize: 28, fontWeight: 700}}>
				Sentences it never saw <span style={{fontFamily: MONO}}>{pct(strict)}</span>
			</Rise>
			<Caption n={n} at={120}>
				The headline result uses the harder number: a new speaker saying a new sentence.
			</Caption>
		</Sheet>
	);
};

const Personal: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const series = data.personalisation;
	const top = Math.max(0.1, ...series.flatMap((s) => s.points.map((p) => p.wer)));
	const maxY = Math.ceil((top * 1.1) / 0.1) * 0.1;
	const x0 = 300;
	const y0 = 230;
	const W = 1100;
	const H = 560;
	const X = (m: number) => x0 + (m / 10) * W;
	const Y = (v: number) => y0 + (1 - v / maxY) * H;
	const colors = [L.clear, L.ink, L.small];
	const first = series[0];
	const n = 6 + (part.total ?? 1);
	return (
		<Sheet>
			<Rise at={4} style={{position: 'absolute', top: 52, left: 100, right: 100, fontSize: 46, lineHeight: 1.12, fontWeight: 800, ...({textWrap: 'balance'} as React.CSSProperties)}}>
				{personalHeadline(series)}
			</Rise>
			<svg style={{position: 'absolute', inset: 0}} width={1920} height={1080}>
				{Array.from({length: Math.round(maxY / 0.1) + 1}, (_, i) => i * 0.1).map((t) => (
					<g key={t}>
						<line x1={x0} x2={x0 + W} y1={Y(t)} y2={Y(t)} stroke={L.grid} strokeWidth={1.5} />
						<text x={x0 - 18} y={Y(t) + 8} textAnchor="end" fontFamily={MONO} fontSize={22} fill={L.muted}>
							{Math.round(t * 100)}%
						</text>
					</g>
				))}
				{[0, 2, 5, 10].map((m) => (
					<text key={m} x={X(m)} y={y0 + H + 40} textAnchor="middle" fontFamily={MONO} fontSize={22} fill={L.muted}>
						{m} min
					</text>
				))}
				{series.map((s, si) => {
					const draw = easeOut((f - 30 - si * 25) / 50);
					const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p.minutes)},${Y(p.wer)}`).join(' ');
					const last = s.points[s.points.length - 1];
					return (
						<g key={s.speaker}>
							<path d={d} fill="none" stroke={colors[si]} strokeWidth={4} pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
							{s.points.map((p, i) => (draw >= i / (s.points.length - 1) - 0.01 ? <circle key={i} cx={X(p.minutes)} cy={Y(p.wer)} r={9} fill={colors[si]} /> : null))}
							{draw > 0.98 ? (
								<text x={X(last.minutes) + 22} y={Y(last.wer) + 9} fontFamily={SANS} fontSize={28} fontWeight={700} fill={colors[si]}>
									{s.speaker}, {s.severity}
								</text>
							) : null}
						</g>
					);
				})}
				<text x={x0 + W / 2} y={y0 + H + 86} textAnchor="middle" fontFamily={SANS} fontSize={24} fill={L.muted}>
					Minutes of the speaker’s own recordings added to training
				</text>
			</svg>
			{first ? (
				<Rise at={150} style={{position: 'absolute', top: 260, left: 1620, width: 260, fontSize: 28, lineHeight: 1.45}}>
					{first.speaker}: <b style={{fontFamily: MONO}}>{pct(first.points[0].wer)}</b> to <b style={{fontFamily: MONO, color: L.clear}}>{pct(first.points[first.points.length - 1].wer)}</b> words wrong
				</Rise>
			) : null}
			<Caption n={n} at={120}>
				Word error rate on each speaker’s unseen sentences, starting from the model that never heard them.
			</Caption>
		</Sheet>
	);
};

const Close: React.FC<SceneProps> = () => (
	<Sheet>
		<Rise at={4} style={{position: 'absolute', top: 200, left: 100, fontSize: 120, fontWeight: 800, letterSpacing: '-0.02em'}}>
			Clear Voice
		</Rise>
		<Rise at={18} style={{position: 'absolute', top: 360, left: 100, fontSize: 40, maxWidth: 1400, lineHeight: 1.4}}>
			Next: test it with speech therapists and with people who have dysarthria, on real dictation.
		</Rise>
		<Rise at={36} style={{position: 'absolute', top: 540, left: 100, fontSize: 32, lineHeight: 1.6}}>
			<div style={{fontWeight: 700}}>{data.team}</div>
			<div style={{fontFamily: MONO}}>{data.repoUrl}</div>
			{data.demoUrl ? <div style={{fontFamily: MONO}}>{data.demoUrl}</div> : null}
		</Rise>
		<Rise at={56} style={{position: 'absolute', top: 820, left: 100, right: 100, fontSize: 22, color: L.muted, lineHeight: 1.5, borderTop: `1.5px solid ${L.grid}`, paddingTop: 18}}>
			<b style={{color: L.ink}}>Data and code.</b> Recordings: TORGO database (Rudzicz, Namasivayam and Wolff, 2012), non-commercial research use. Models: OpenAI Whisper. Code, splits and results are in the repository. Research prototype, not a medical device. Built with help from Claude (Anthropic).
		</Rise>
	</Sheet>
);

export const LabFilm: React.FC = () => (
	<Film
		ground={L.ground}
		scenes={{prologue: Prologue, epilogue: Epilogue, figures: PaperFigures, title: Title, hook: Hook, problem: Problem, demo: Demo, how: How, results: Results, honest: Honest, personal: Personal, close: Close}}
		music={{file: 'sfx/music.wav', volume: 0.2}}
	/>
);

