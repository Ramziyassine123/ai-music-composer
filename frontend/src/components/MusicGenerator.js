import React, { useState } from 'react';
import axios from 'axios';
import { playMidiData } from '../utils/audioUtils';

const MusicGenerator = () => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedMusic, setGeneratedMusic] = useState(null);
    const [settings, setSettings] = useState({
        length: 32,
        temperature: 1.0
    });

    const generateMusic = async () => {
        setIsGenerating(true);

        try {
            const response = await axios.post('/generate', settings, {
                responseType: 'blob'
            });

            setGeneratedMusic(response.data);

        } catch (error) {
            console.error('Generation failed:', error);
            alert('Failed to generate music. Make sure the backend is running!');
        } finally {
            setIsGenerating(false);
        }
    };

    const playMusic = async () => {
        if (generatedMusic) {
            try {
                await playMidiData(generatedMusic);
            } catch (error) {
                console.error('Playback failed:', error);
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
                    <label>Length (notes):</label>
                    <input
                        type="range"
                        min="16"
                        max="64"
                        value={settings.length}
                        onChange={(e) => setSettings({...settings, length: parseInt(e.target.value)})}
                    />
                    <span>{settings.length}</span>
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
                            ▶️ Play
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