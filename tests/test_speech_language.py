"""Piper voice selection, including unaccented French replies."""

from hyper_ambient_maxi.cli import main
from hyper_ambient_maxi.ears.local_speech import speech_language, _voice_key


def test_unaccented_french_reply_selects_the_french_voice():
    assert speech_language("Votre chien s'appelle Biscuit.") == "fr"
    assert speech_language("Votre chien s'appelle Biscuit") == "fr"
    assert _voice_key("Votre chien s'appelle Biscuit.") == "fr"
    assert speech_language("Je vais m'en souvenir. User works at Airbus.") == "fr"


def test_english_reply_stays_english():
    assert speech_language("Your dog's name is Biscuit.") == "en"
    assert speech_language("I'll remember that. User's dog is called Pixel.") == "en"


def test_preferred_language_wins_unless_the_text_is_clearly_the_other():
    assert speech_language("Biscuit.", preferred="fr") == "fr"
    assert speech_language("Biscuit.", preferred="en") == "en"
    assert speech_language("Your dog's name is Biscuit.", preferred="fr") == "en"
    assert speech_language("Votre chien s'appelle Biscuit.", preferred="en") == "fr"
    assert speech_language("Hello.", preferred="fr-FR") == "fr"


def test_download_models_help_mentions_piper_voices():
    # argparse writes help to stdout and exits; call the flag via main's parser
    # through a subprocess-free path: the option help string.
    import hyper_ambient_maxi.cli as cli
    import argparse

    parser = None
    # Recreate the parser the same way main() does, by inspecting --help text
    # from a captured main() is awkward; parse the module source via argparse
    # by invoking main with --help through SystemExit.
    import io
    from contextlib import redirect_stdout, redirect_stderr
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    help_text = stdout.getvalue()
    assert "--download-models" in help_text
    assert "Whisper" in help_text or "whisper" in help_text.lower()
    assert "Piper" in help_text
