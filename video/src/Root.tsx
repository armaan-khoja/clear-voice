import {Composition} from 'remotion';
import {FPS, totalFrames} from './shared/timeline';
import {ExamFilm} from './styles/exam/Exam';
import {LabFilm} from './styles/lab/Lab';
import {ProofFilm} from './styles/proof/Proof';

// Three looks on one timeline: the team records one voiceover and it fits whichever is chosen.
export const Root: React.FC = () => (
	<>
		<Composition id="Exam" component={ExamFilm} durationInFrames={totalFrames()} fps={FPS} width={1920} height={1080} />
		<Composition id="Lab" component={LabFilm} durationInFrames={totalFrames()} fps={FPS} width={1920} height={1080} />
		<Composition id="Proof" component={ProofFilm} durationInFrames={totalFrames()} fps={FPS} width={1920} height={1080} />
	</>
);
