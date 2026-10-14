"""
SignBridge AI - Text-to-Speech (TTS) Engine
Asynchronous, non-blocking speech synthesizer using macOS native 'say'
or cross-platform pyttsx3.
"""

import os
import platform
import subprocess
import threading
from typing import Optional


class TTSEngine:
    """
    Manages speech playback in a background thread so the camera stream never lags.
    """

    def __init__(self, voice_rate: int = 175):
        self.system = platform.system()
        self.voice_rate = voice_rate
        self._lock = threading.Lock()
        self._is_speaking = False

    def speak(self, text: str) -> None:
        """
        Speaks text asynchronously without blocking the caller.

        Args:
            text: Text string to pronounce.
        """
        clean_text = text.strip()
        if not clean_text:
            return

        thread = threading.Thread(
            target=self._speak_worker,
            args=(clean_text,),
            daemon=True
        )
        thread.start()

    def _speak_worker(self, text: str) -> None:
        """Executes system-specific speech command."""
        with self._lock:
            self._is_speaking = True
            try:
                if self.system == "Darwin":
                    # macOS native speech synthesizer
                    subprocess.run(
                        ["/usr/bin/say", "-r", str(self.voice_rate), text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                else:
                    # Windows / Linux fallback with pyttsx3
                    try:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty("rate", self.voice_rate)
                        engine.say(text)
                        engine.runAndWait()
                        engine.stop()
                    except Exception as err:
                        print(f"[TTSEngine] pyttsx3 error: {err}")
            except Exception as e:
                print(f"[TTSEngine] Failed to speak: {e}")
            finally:
                self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
