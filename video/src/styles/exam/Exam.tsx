import {Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import type {Clip} from '../../data';
import {BIGGER, data, STOCK, wordCer} from '../../data';
import {easeOut, pct} from '../../shared/audio';
import type {SceneProps} from '../../shared/Film';
import {Film, Watermark} from '../../shared/Film';
import {Epilogue, Prologue} from '../../shared/Human';
import {clipEnd} from '../../shared/timeline';
import {Answer, Box, CANDIDATES, COL, E, Fade, HAND, LEFT, Page, PRINT, Question, Tick, TYPE, WaveFigure} from './parts';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const SEVERITY: Record<string, string> = {mild: 'mild', moderate: 'moderate', severe: 'severe', 'moderate-severe': 'moderate to severe'};

// When each candidate starts writing: one after another, each after the examiner has marked the last.
const schedule = (clip: Clip, start: number, fps: number, cps: number) => {
	const out: number[] = [];
	let t = start;
	for (const c of CANDIDATES) {
		out.push(t);
		const s = clip.systems[c.key];
		const text = s ? s.tokens.map((x) => x.w).join(' ') : '';
		const bad = s ? s.tokens.filter((x) => x.tag !== 'ok').length : 0;
		t += Math.ceil((text.length / cps) * fps) + 6 + bad * 5 + 22;
	}
	return {starts: out, end: t};
};

const AnswerSheet: React.FC<{clip: Clip; playAt: number; n: number; prompt: React.ReactNode; marks: number; cps: number; figCaption: string}> = ({
	clip,
	playAt,
	n,
	prompt,
	marks,
	cps,
	figCaption,
}) => {
	const {fps} = useVideoConfig();
	const end = clipEnd({clip, playAt});
	const {starts, end: marked} = schedule(clip, end + 10, fps, cps);
	const refWords = clip.ref.split(' ').length;
	return (
		<>
			<Question n={n} marks={marks} top={130} at={4}>
				{prompt}
			</Question>
			<WaveFigure file={clip.file} duration={clip.duration} playFrom={playAt} top={236} caption={figCaption} height={130} />
			{CANDIDATES.map((c, i) =>
				clip.systems[c.key] ? (
					<Answer key={c.key} cand={c} tokens={clip.systems[c.key].tokens} refWords={refWords} top={455 + i * 112} at={starts[i]} cps={cps} size={38} />
				) : null,
			)}
			<Box title="Mark scheme" at={marked} style={{top: 800, left: LEFT, width: COL - LEFT - 30}}>
				<div style={{fontFamily: TYPE, fontSize: 36}}>{clip.ref}</div>
			</Box>
		</>
	);
};

const Title: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	const digits = '0421';
	return (
		<Page n={1} examinerColumn={false} turnOver={false}>
			<div style={{position: 'absolute', top: 150, left: LEFT}}>
				<div style={{fontSize: 92, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1}}>Clear Voice</div>
				<div style={{fontSize: 30, marginTop: 14, color: E.soft}}>Speech recognition for people with dysarthria</div>
			</div>
			<div style={{position: 'absolute', top: 150, right: 120, display: 'flex', flexDirection: 'column', gap: 18, fontSize: 22}}>
				{[
					['Centre number', '7300'],
					['Candidate number', digits],
				].map(([label, value], r) => (
					<div key={label} style={{display: 'flex', alignItems: 'center', gap: 16, justifyContent: 'flex-end'}}>
						<span style={{color: E.soft}}>{label}</span>
						<div style={{display: 'flex'}}>
							{value.split('').map((d, i) => (
								<div key={i} style={{width: 54, height: 60, border: `1.5px solid ${E.rule}`, marginLeft: -1.5, display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: HAND, fontSize: 38, color: E.blue}}>
									{f > 30 + r * 16 + i * 4 ? d : ''}
								</div>
							))}
						</div>
					</div>
				))}
			</div>
			<div style={{position: 'absolute', top: 390, left: LEFT, right: 120, height: 2, background: E.rule}} />
			<Fade at={8} style={{position: 'absolute', top: 430, left: LEFT}}>
				<div style={{fontSize: 64, fontWeight: 700, letterSpacing: '-0.015em'}}>Paper 1: Unseen speakers, unseen sentences</div>
				<div style={{fontSize: 30, marginTop: 12, color: E.soft}}>Time allowed: 2 minutes</div>
			</Fade>
			<Fade at={30} style={{position: 'absolute', top: 600, left: LEFT, width: 760, fontSize: 30, lineHeight: 1.5}}>
				<div style={{fontWeight: 700, marginBottom: 8}}>Candidates</div>
				{CANDIDATES.map((c) => (
					<div key={c.key}>
						{c.letter} <span style={{color: c.ink, fontWeight: 700}}>{c.name}</span> <span style={{color: E.soft}}>({c.note})</span>
					</div>
				))}
			</Fade>
			<Fade at={44} style={{position: 'absolute', top: 600, left: 960, width: 840, fontSize: 30, lineHeight: 1.5}}>
				<div style={{fontWeight: 700, marginBottom: 8}}>Instructions</div>
				<div>Answer every question.</div>
				<div>Each question is a recording of a person with dysarthria.</div>
				<div>No candidate has heard these speakers or these sentences.</div>
				<div>The examiner marks every word.</div>
			</Fade>
			<Fade at={70} style={{position: 'absolute', bottom: 150, left: 0, right: 0, textAlign: 'center', fontSize: 34, fontWeight: 800}}>
				Do not open this paper until you are told to do so.
			</Fade>
			<Watermark color={E.red} font={PRINT} />
		</Page>
	);
};

const Hook: React.FC<SceneProps> = ({part}) => {
	const clip = part.clip!;
	return (
		<Page n={2}>
			<AnswerSheet
				clip={clip}
				playAt={part.playAt!}
				n={1}
				marks={clip.ref.split(' ').length}
				cps={55}
				prompt="Listen to the recording. Write down exactly what the speaker said."
				figCaption={`Fig. 1.1  Speaker ${clip.speaker}, ${SEVERITY[clip.severity]} dysarthria, ${clip.duration.toFixed(1)} seconds`}
			/>
		</Page>
	);
};

const Problem: React.FC<SceneProps> = () => {
	const f = useCurrentFrame();
	const stock = data.headline.find((b) => b.name === STOCK)!.value;
	const ring = interpolate(f - 175, [0, 14], [0, 1], clamp);
	return (
		<Page n={3} examinerColumn={false}>
			<div style={{position: 'absolute', top: 128, left: LEFT, fontSize: 40, fontWeight: 800}}>Insert</div>
			<Box title="Source A  Dysarthria" at={10} style={{top: 210, left: LEFT, width: 1620}}>
				<div style={{fontSize: 38, lineHeight: 1.45, maxWidth: 1480}}>
					Dysarthria makes speech slow, slurred or quiet because the muscles used for speaking are weak or hard to control. It can come with cerebral palsy, ALS, a stroke or Parkinson’s disease.
				</div>
			</Box>
			<Box title="Source B  How stock speech recognition does" at={70} style={{top: 520, left: LEFT, width: 1620}}>
				<div style={{fontSize: 38, lineHeight: 1.45, maxWidth: 1480, position: 'relative'}}>
					Dictation and live captions are trained mostly on typical voices. On sentences from speakers with dysarthria, stock Whisper gets{' '}
					<span style={{position: 'relative', fontWeight: 800, fontSize: 54, lineHeight: 0, whiteSpace: 'nowrap', margin: '0 16px 0 20px'}}>
						{pct(stock)}
						<svg style={{position: 'absolute', left: -22, top: -2, overflow: 'visible'}} width={150} height={80}>
							<path d="M128 20 C 122 0, 22 -2, 8 24 C -2 46, 30 74, 82 70 C 132 66, 150 40, 120 14" fill="none" stroke={E.red} strokeWidth={3} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - ring} />
						</svg>
					</span>{' '}
					of the words wrong.
				</div>
				<div style={{fontSize: 24, color: E.soft, marginTop: 14}}>Word error rate, TORGO database, speakers and sentences the model was not trained on.</div>
			</Box>
		</Page>
	);
};

const Demo: React.FC<SceneProps> = ({part}) => {
	const clip = part.clip!;
	const n = 2 + (part.index ?? 0);
	return (
		<Page n={4 + (part.index ?? 0)}>
			<AnswerSheet
				clip={clip}
				playAt={part.playAt!}
				n={n}
				marks={clip.ref.split(' ').length}
				cps={70}
				prompt={`Speaker ${clip.speaker} has ${SEVERITY[clip.severity]} dysarthria. Write down what they said.`}
				figCaption={`Fig. ${n}.1  Chosen at random from the unseen test set, ${clip.duration.toFixed(1)} seconds`}
			/>
		</Page>
	);
};

const How: React.FC<SceneProps> = ({part}) => {
	const n = 4 + (part.total ?? 1);
	const cell = (label: string, body: string, at: number, ink = E.ink) => (
		<Fade at={at} style={{borderTop: `1.5px solid ${E.rule}`, padding: '18px 0', display: 'flex', gap: 28, fontSize: 32, lineHeight: 1.4}}>
			<div style={{width: 230, color: E.soft}}>{label}</div>
			<div style={{flex: 1, color: ink}}>{body}</div>
		</Fade>
	);
	return (
		<Page n={n}>
			<div style={{position: 'absolute', top: 128, left: LEFT, fontSize: 48, fontWeight: 800}}>How this paper was set</div>
			<div style={{position: 'absolute', top: 210, left: LEFT, width: 700}}>
				<Fade at={10} style={{fontSize: 34, fontWeight: 700, marginBottom: 14}}>Practice (training)</Fade>
				{cell('Speakers', '4 people with dysarthria and the typical speakers', 18)}
				{cell('Sentences', '70% of the prompts', 26)}
				{cell('Model', `OpenAI ${STOCK}, fine-tuned on one GPU`, 34)}
				{cell('Checkpoint', 'the other 10% of the prompts, only to pick when to stop', 42)}
			</div>
			<div style={{position: 'absolute', top: 210, left: 900, width: 700}}>
				<Fade at={60} style={{fontSize: 34, fontWeight: 700, marginBottom: 14}}>This paper (test)</Fade>
				{cell('Speakers', '4 different people with dysarthria', 68)}
				{cell('Sentences', '20% of the prompts, never used in practice', 76)}
				{cell('Then', 'the groups swap, so every speaker sits the paper once', 84)}
			</div>
			<Box title="Security" at={150} style={{top: 760, left: LEFT, width: COL - LEFT - 30}}>
				<div style={{fontSize: 32, lineHeight: 1.45}}>
					Near-identical prompts, such as “as I choose” and “as I chose”, were kept on the same side, so no question can be answered from memory. {data.nRecordings.toLocaleString('en-US')} recordings, {data.nDysarthric} speakers with dysarthria.
				</div>
			</Box>
			<div style={{position: 'absolute', top: 830, left: COL + 40}}>
				<Tick at={185} size={54} />
			</div>
		</Page>
	);
};

const Results: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const n = 5 + (part.total ?? 1);
	const bars = data.headline;
	const words = Object.fromEntries(data.headlineWords.map((b) => [b.name, b.value]));
	const max = Math.max(...bars.map((b) => b.hi ?? b.value)) * 1.05;
	const ours = bars.find((b) => b.name === 'Clear Voice')!.value;
	const stock = bars.find((b) => b.name === STOCK)!.value;
	const large = bars.find((b) => b.name === 'whisper-large-v3')?.value ?? Infinity;
	const verdict =
		ours < Math.min(stock, large)
			? 'Clear Voice made the fewest mistakes.'
			: ours < stock
				? (wordCer('Clear Voice') ?? 9) < Math.min(wordCer(STOCK) ?? 9, wordCer('whisper-large-v3') ?? 9)
					? `Fewer errors than its base model, though the ranges overlap; the ${BIGGER} bigger model still leads. On single words, Clear Voice is best.`
					: `Fewer errors than its base model, though the ranges overlap; the ${BIGGER} bigger model still made fewer mistakes.`
				: 'Clear Voice did not beat stock Whisper on this paper.';
	const W = 520;
	return (
		<Page n={n}>
			<div style={{position: 'absolute', top: 128, left: LEFT, fontSize: 48, fontWeight: 800}}>Results</div>
			<Fade at={6} style={{position: 'absolute', top: 196, left: LEFT, fontSize: 30, color: E.soft}}>
				Speakers with dysarthria the models never heard, saying sentences they never heard.
			</Fade>
			<div style={{position: 'absolute', top: 270, left: LEFT, width: COL - LEFT - 30, fontSize: 30}}>
				<div style={{display: 'flex', borderBottom: `2px solid ${E.rule}`, paddingBottom: 12, fontWeight: 700}}>
					<div style={{width: 380}}>Candidate</div>
					<div style={{width: W + 300}}>Sentences: words wrong (95% range)</div>
					<div style={{flex: 1, textAlign: 'right'}}>Single words: letters wrong</div>
				</div>
				{bars.map((b, i) => {
					const c = CANDIDATES.find((x) => x.key === b.name)!;
					const t = easeOut((f - 30 - i * 16) / 24);
					return (
						<div key={b.name} style={{display: 'flex', alignItems: 'center', borderBottom: `1.5px solid ${E.hair}`, padding: '26px 0', opacity: t > 0 ? 1 : 0}}>
							<div style={{width: 380}}>
								{c.letter} <span style={{fontWeight: 700, color: c.ink}}>{c.name}</span>
								<div style={{fontSize: 22, color: E.soft}}>{c.note}</div>
							</div>
							<div style={{width: W + 300, display: 'flex', alignItems: 'center', gap: 20}}>
								<svg width={W} height={40} style={{flexShrink: 0, overflow: 'visible'}}>
									<rect x={0} y={8} width={(b.value / max) * W * t} height={24} fill={c.ink} />
									{b.lo !== undefined && b.hi !== undefined ? (
										<g opacity={t}>
											{[{stroke: E.paper, w: 5}, {stroke: E.ink, w: 2}].map((s) => (
												<g key={s.w} stroke={s.stroke} strokeWidth={s.w}>
													<line x1={(b.lo / max) * W} x2={(b.hi / max) * W} y1={20} y2={20} />
													<line x1={(b.lo / max) * W} x2={(b.lo / max) * W} y1={12} y2={28} />
													<line x1={(b.hi / max) * W} x2={(b.hi / max) * W} y1={12} y2={28} />
												</g>
											))}
										</g>
									) : null}
								</svg>
								<div style={{fontWeight: 800, fontSize: 38, fontVariantNumeric: 'tabular-nums'}}>{pct(b.value)}</div>
								{b.lo !== undefined && b.hi !== undefined ? (
									<div style={{fontSize: 24, color: E.soft, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap'}}>
										{pct(b.lo)} to {pct(b.hi)}
									</div>
								) : null}
							</div>
							<div style={{flex: 1, textAlign: 'right', fontWeight: 700, fontSize: 34, fontVariantNumeric: 'tabular-nums'}}>{words[b.name] !== undefined ? pct(words[b.name]) : ''}</div>
						</div>
					);
				})}
			</div>
			<Fade at={120} style={{position: 'absolute', top: 760, left: LEFT, width: 1400, fontFamily: HAND, fontSize: 46, color: E.red, transform: 'rotate(-1deg)'}}>
				{verdict}
			</Fade>
		</Page>
	);
};

const Honest: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const n = 6 + (part.total ?? 1);
	const seen = data.leakage.find((b) => /seen sentence/i.test(b.name))!;
	const strict = data.leakage.find((b) => /new speaker, new/i.test(b.name))!;
	const arrow = interpolate(f - 150, [0, 16], [0, 1], clamp);
	const row = (label: string, v: number, at: number, bold: boolean) => (
		<Fade at={at} style={{display: 'flex', justifyContent: 'space-between', borderBottom: `1.5px solid ${E.hair}`, padding: '28px 0', fontSize: 38, fontWeight: bold ? 800 : 500}}>
			<span>{label}</span>
			<span style={{fontVariantNumeric: 'tabular-nums'}}>{pct(v)}</span>
		</Fade>
	);
	return (
		<Page n={n}>
			<div style={{position: 'absolute', top: 128, left: LEFT, fontSize: 48, fontWeight: 800}}>Examiner’s report</div>
			<Fade at={8} style={{position: 'absolute', top: 210, left: LEFT, width: 1300, fontSize: 34, lineHeight: 1.45}}>
				The same model, the same new speakers. Only the sentences change.
			</Fade>
			<div style={{position: 'absolute', top: 320, left: LEFT, width: 1100}}>
				{row('Sentences it practised on', seen.value, 30, false)}
				{row('Sentences it never saw', strict.value, 60, true)}
			</div>
			<Fade at={110} style={{position: 'absolute', top: 620, left: LEFT + 40, width: 1100, fontFamily: HAND, fontSize: 50, color: E.red, transform: 'rotate(-1.5deg)'}}>
				{strict.value > seen.value
					? `Practised sentences flatter it by ${Math.round((strict.value - seen.value) * 100)} points. Report the unseen score.`
					: 'Practised sentences gave it no advantage here. Report the unseen score.'}
			</Fade>
			<svg style={{position: 'absolute', left: LEFT + 1120, top: 470, overflow: 'visible'}} width={200} height={220}>
				<path d="M40 200 C 120 170, 130 90, 30 20" fill="none" stroke={E.red} strokeWidth={3} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - arrow} />
				<path d="M30 20 l 26 4 M30 20 l 6 26" stroke={E.red} strokeWidth={3} strokeLinecap="round" opacity={arrow > 0.95 ? 1 : 0} />
			</svg>
		</Page>
	);
};

const Personal: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const n = 7 + (part.total ?? 1);
	const series = data.personalisation;
	const W = 980;
	const H = 540;
	const x0 = 290;
	const y0 = 300;
	const top = Math.max(0.1, ...series.flatMap((s) => s.points.map((p) => p.wer)));
	const maxY = Math.ceil((top * 1.1) / 0.1) * 0.1;
	const X = (m: number) => x0 + (m / 10) * W;
	const Y = (v: number) => y0 + (1 - v / maxY) * H;
	const bands = Math.max(1, Math.round(maxY / 0.1)); // one major grid line per 10%
	const inks = [E.blue, E.ink, E.pencil, E.red];
	const first = series[0];
	return (
		<Page n={n}>
			<Question n={2 + (part.total ?? 1)} top={128} at={2}>
				How much does a speaker’s own voice help? The model is trained further on 2, 5 and 10 minutes of one person’s recordings.
			</Question>
			<svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
				<defs>
					<pattern id="minor" width={W / 50} height={H / (bands * 5)} patternUnits="userSpaceOnUse" x={x0} y={y0}>
						<path d={`M ${W / 50} 0 L 0 0 0 ${H / (bands * 5)}`} fill="none" stroke="#c8d6e4" strokeWidth={0.8} />
					</pattern>
					<pattern id="major" width={W / 10} height={H / bands} patternUnits="userSpaceOnUse" x={x0} y={y0}>
						<rect width={W / 10} height={H / bands} fill="url(#minor)" />
						<path d={`M ${W / 10} 0 L 0 0 0 ${H / bands}`} fill="none" stroke="#97b2cc" strokeWidth={1.4} />
					</pattern>
				</defs>
				<rect x={x0} y={y0} width={W} height={H} fill="url(#major)" stroke="#97b2cc" strokeWidth={1.4} />
				{Array.from({length: Math.round(maxY / 0.1) + 1}, (_, i) => i * 0.1).map((t) => (
					<text key={t} x={x0 - 16} y={Y(t) + 9} textAnchor="end" fontFamily={PRINT} fontSize={24} fill={E.soft}>
						{Math.round(t * 100)}%
					</text>
				))}
				{[0, 2, 5, 10].map((m) => (
					<text key={m} x={X(m)} y={y0 + H + 40} textAnchor="middle" fontFamily={PRINT} fontSize={24} fill={E.soft}>
						{m}
					</text>
				))}
				<text x={x0 + W / 2} y={y0 + H + 84} textAnchor="middle" fontFamily={PRINT} fontSize={26} fill={E.ink}>
					Minutes of the speaker’s own recordings
				</text>
				<text x={x0 - 90} y={y0 - 24} fontFamily={PRINT} fontSize={26} fill={E.ink}>
					Words wrong
				</text>
				{series.map((s, si) => {
					const pts = s.points;
					const draw = easeOut((f - 40 - si * 30) / 50);
					const d = pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.minutes)},${Y(p.wer)}`).join(' ');
					const last = pts[pts.length - 1];
					return (
						<g key={s.speaker}>
							<path d={d} fill="none" stroke={inks[si]} strokeWidth={3} pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
							{pts.map((p, i) => {
								const show = draw >= i / Math.max(1, pts.length - 1) - 0.01;
								return show ? (
									<g key={i} stroke={inks[si]} strokeWidth={3} strokeLinecap="round">
										<line x1={X(p.minutes) - 10} y1={Y(p.wer) - 10} x2={X(p.minutes) + 10} y2={Y(p.wer) + 10} />
										<line x1={X(p.minutes) - 10} y1={Y(p.wer) + 10} x2={X(p.minutes) + 10} y2={Y(p.wer) - 10} />
									</g>
								) : null;
							})}
							{draw > 0.98 ? (
								<text x={X(last.minutes) + 24} y={Y(last.wer) + 9} fontFamily={PRINT} fontSize={26} fontWeight={700} fill={inks[si]}>
									{s.speaker}, {s.severity}
								</text>
							) : null}
						</g>
					);
				})}
			</svg>
			{first ? (
				<Fade at={150} style={{position: 'absolute', top: 330, left: COL + 18, width: 150, fontFamily: HAND, fontSize: 38, lineHeight: 1.15, color: E.red, textAlign: 'center'}}>
					{first.speaker}
					<div style={{fontSize: 44, fontWeight: 700}}>{pct(first.points[0].wer)}</div>
					<svg width={40} height={60} viewBox="0 0 40 60">
						<path d="M20 4 C 18 20, 22 36, 20 52 M10 42 L20 54 L31 41" fill="none" stroke={E.red} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
					</svg>
					<div style={{fontSize: 44, fontWeight: 700}}>{pct(first.points[first.points.length - 1].wer)}</div>
					<div style={{fontSize: 30}}>after {first.points[first.points.length - 1].minutes} min</div>
				</Fade>
			) : null}
		</Page>
	);
};

// Three figures from the team's research paper, shown as an exam insert; the examiner rings what matters in each.
export const PAPER_FIGURES = [
	{file: 'paper/fig2-split.png', w: 1150, h: 728, ring: [0.74, 0.55, 0.99, 0.98], note: 'Only this cell counts.', caption: 'Figure 2: where each recording goes. The headline uses only the dark cell, new speakers reading new prompts.'},
	{file: 'paper/fig4-headline.png', w: 1627, h: 585, ring: [0.885, 0.52, 0.985, 0.8], note: 'Lowest on single words: 39%', caption: 'Figure 4: the headline, all 690 test clips. Whiskers are 95% ranges over the eight speakers.'},
	{file: 'paper/fig5-models.png', w: 1629, h: 584, ring: [0.68, 0.15, 0.84, 0.86], note: 'Fine-tuning cut single-word errors for every size', caption: 'Figure 5: each Whisper size before and after fine-tuning, on the same test.'},
];
const FIG_LEN = 135;

const Figures: React.FC<SceneProps> = ({part}) => {
	const f = useCurrentFrame();
	const last = PAPER_FIGURES.length - 1;
	return (
		<Page n={8 + (part.total ?? 1)}>
			<Fade at={4} style={{position: 'absolute', top: 116, left: LEFT, fontSize: 40, fontWeight: 800}}>
				Insert: figures from our write-up
			</Fade>
			<Fade at={10} style={{position: 'absolute', top: 172, left: LEFT, fontSize: 22, color: E.soft}}>
				From our research paper, “Clear Voice: Fine-Tuning Whisper for Dysarthric Speech”.
			</Fade>
			{PAPER_FIGURES.map((g, i) => {
				const a = 22 + i * FIG_LEN;
				const b = a + FIG_LEN;
				const o = i === last ? interpolate(f, [a, a + 12], [0, 1], clamp) : interpolate(f, [a, a + 12, b, b + 12], [0, 1, 1, 0], clamp);
				const sc = Math.min(1440 / g.w, 560 / g.h);
				const w = g.w * sc;
				const h = g.h * sc;
				const top = 228;
				const [x0, y0, x1, y1] = g.ring;
				const rx = ((x1 - x0) * w) / 2 + 10;
				const ry = ((y1 - y0) * h) / 2 + 10;
				const cx = ((x0 + x1) / 2) * w;
				const cy = ((y0 + y1) / 2) * h;
				const per = Math.PI * (3 * (rx + ry) - Math.sqrt((3 * rx + ry) * (rx + 3 * ry)));
				const draw = easeOut((f - a - 32) / 18);
				return (
					<div key={g.file} style={{position: 'absolute', inset: 0, opacity: o}}>
						<Img src={staticFile(g.file)} style={{position: 'absolute', left: LEFT, top, width: w, height: h, outline: `1.5px solid ${E.hair}`, outlineOffset: 0}} />
						<svg style={{position: 'absolute', left: LEFT, top, overflow: 'visible'}} width={w} height={h}>
							<ellipse cx={cx} cy={cy} rx={rx} ry={ry} fill="none" stroke={E.red} strokeWidth={4} strokeDasharray={per} strokeDashoffset={per * (1 - draw)} transform={`rotate(-4 ${cx} ${cy})`} />
						</svg>
						<div style={{position: 'absolute', left: LEFT, top: top + h + 26, width: COL - LEFT - 60, fontSize: 24, lineHeight: 1.4, color: E.soft}}>{g.caption}</div>
						<div style={{position: 'absolute', left: COL + 16, top: 380, width: 240, fontFamily: HAND, fontSize: 34, lineHeight: 1.2, color: E.red, opacity: interpolate(f, [a + 48, a + 60], [0, 1], clamp)}}>{g.note}</div>
					</div>
				);
			})}
		</Page>
	);
};

const Close: React.FC<SceneProps> = ({part}) => {
	return (
		<Page n={9 + (part.total ?? 1)} examinerColumn={false} turnOver={false}>
			<Fade at={6} style={{position: 'absolute', top: 300, left: 0, right: 0, textAlign: 'center', fontSize: 44, fontWeight: 800, letterSpacing: '0.06em'}}>
				END OF PAPER
			</Fade>
			<Fade at={30} style={{position: 'absolute', top: 400, left: 0, right: 0, textAlign: 'center', fontSize: 34, lineHeight: 1.5}}>
				Next: testing with speech therapists, on real dictation.
				<div style={{marginTop: 26, fontWeight: 700}}>{data.team}</div>
				<div style={{marginTop: 8, color: E.soft}}>
					{data.repoUrl}
					{data.demoUrl ? `   ·   try it: ${data.demoUrl}` : ''}
				</div>
			</Fade>
			<Fade at={60} style={{position: 'absolute', bottom: 150, left: LEFT, right: 120, fontSize: 21, lineHeight: 1.5, color: E.soft, borderTop: `1.5px solid ${E.hair}`, paddingTop: 18}}>
				Acknowledgements. Recordings from the TORGO database (Rudzicz, Namasivayam and Wolff, 2012), used for non-commercial research. Baseline models: OpenAI Whisper. Clear Voice is a research prototype, not a medical device. Built with help from Claude (Anthropic).
			</Fade>
		</Page>
	);
};

// After the dark prologue the cover page fades up instead of appearing at once.
const TitleIn: React.FC<SceneProps> = (props) => {
	const f = useCurrentFrame();
	return (
		<div style={{position: 'absolute', inset: 0, opacity: interpolate(f, [0, 20], [0, 1], clamp)}}>
			<Title {...props} />
		</div>
	);
};

export const ExamFilm: React.FC = () => (
	<Film
		ground={E.paper}
		scenes={{prologue: Prologue, epilogue: Epilogue, figures: Figures, title: TitleIn, hook: Hook, problem: Problem, demo: Demo, how: How, results: Results, honest: Honest, personal: Personal, close: Close}}
		music={{file: 'sfx/music.wav', volume: 0.16}}
		turn={{file: 'sfx/paper.wav', volume: 0.5, offset: -2}}
	/>
);
