import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { playMidiData, stopPlayback } from '../utils/audioUtils';

// Each generated step is 1/8 of a second (see backend midi_utils.py)
const STEP_SECONDS = 0.125;

const MusicGenerator = () => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedMusic, setGeneratedMusic] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [settings, setSettings] = useState({
        length: 128,
        temperature: 1.0
    });

    // Stop any playing audio when the component unmounts
    useEffect(() => stopPlayback, []);

    const generateMusic = async () => {
        setIsGenerating(true);
        stopPlayback();
        setIsPlaying(false);

        try {
            const response = await axios.post('/generate', settings, {
                responseType: 'blob'
            });

            setGeneratedMusic(response.data);

        } catch (error) {
            console.error('Generation failed:', error);
            let detail = 'Make sure the backend is running!';
            // The error body is a Blob because of responseType: 'blob'
            if (error.response && error.response.data instanceof Blob) {
                try {
                    detail = JSON.parse(await error.response.data.text()).detail || detail;
                } catch (e) { /* not JSON */ }
            }
            alert(`Failed to generate music: ${detail}`);
        } finally {
            setIsGenerating(false);
        }
    };

    const playMusic = async () => {
        if (isPlaying) {
            stopPlayback();
            setIsPlaying(false);
            return;
        }
        if (generatedMusic) {
            try {
                setIsPlaying(true);
                await playMidiData(generatedMusic, () => setIsPlaying(false));
            } catch (error) {
                console.error('Playback failed:', error);
                setIsPlaying(false);
                alert('Playback failed. Please try again.');
            }
        }
    };

    const downloadMusic = () => {
        if (generatedMusic) {
            const url = URL.createObjectURL(generatedMusic);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'generated_music.mid';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    };

    return (
        <div className="music-generator">
            <div className="controls">
                <div className="control-group">
                    <label>Length:</label>
                    <input
                        type="range"
                        min="32"
                        max="256"
                        step="16"
                        value={settings.length}
                        onChange={(e) => setSettings({...settings, length: parseInt(e.target.value)})}
                    />
                    <span>{settings.length * STEP_SECONDS}s</span>
                </div>

                <div className="control-group">
                    <label>Creativity:</label>
                    <input
                        type="range"
                        min="0.1"
                        max="2.0"
                        step="0.1"
                        value={settings.temperature}
                        onChange={(e) => setSettings({...settings, temperature: parseFloat(e.target.value)})}
                    />
                    <span>{settings.temperature}</span>
                </div>

                <button
                    onClick={generateMusic}
                    disabled={isGenerating}
                    className="generate-btn"
                >
                    {isGenerating ? 'Generating...' : '🎼 Generate Music'}
                </button>
            </div>

            {generatedMusic && (
                <div className="music-controls">
                    <h3>🎉 Music Generated!</h3>
                    <div className="playback-controls">
                        <button onClick={playMusic} className="play-btn">
                            {isPlaying ? '⏹️ Stop' : '▶️ Play'}
                        </button>
                        <button onClick={downloadMusic} className="download-btn">
                            💾 Download MIDI
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MusicGenerator;
