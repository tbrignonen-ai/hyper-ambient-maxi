"""French slots share keys with the English ones."""

from datetime import date, timedelta

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.dates import resolve_fr_when
from hyper_ambient_maxi.memory.extract import extract
from hyper_ambient_maxi.memory.store import MemoryStore


def test_french_diagnosis_job_and_rename_use_the_english_slots():
    dog = extract("Mon chien s'appelle Pixel.")
    assert dog.facts[0].key == "fact:dog-name"
    assert dog.facts[0].value == "Pixel"
    health = extract(
        "Ma mère a été diagnostiquée d'un cancer du sein la semaine dernière, et j'ai peur."
    )
    assert health.facts[0].key == "health:mum-diagnosis"
    assert health.facts[0].value == "cancer du sein"
    assert any(thread.id == "thread:health-mum" for thread in health.threads)
    job = extract("Je viens de décrocher un poste chez Airbus.")
    assert job.facts[0].key == "job:employer"
    assert job.facts[0].value == "Airbus"
    renamed = extract("Je l'ai renommé Biscuit.")
    assert renamed.facts[0].value == "Biscuit"
    assert extract("Souviens-toi de ça.").rejected_pronoun
    assert extract("Souviens-toi de ça.").facts == []


def test_french_turn_recalls_in_french_and_supersedes(tmp_path):
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    scared = run_turn(store, brain, "Ma mère a été diagnostiquée d'un cancer du sein, et j'ai peur.")
    assert scared.memory_ok
    assert "désolé" in scared.reply.lower() or "desole" in scared.reply.lower()
    assert store.slot("health:mum-diagnosis").value == "cancer du sein"
    run_turn(store, brain, "Mon chien s'appelle Pixel.")
    renamed = run_turn(store, brain, "Je l'ai renommé Biscuit.")
    assert "remplace" in renamed.reply
    asked = run_turn(store, brain, "Comment s'appelle mon chien ?")
    assert asked.reply == "Ton chien s'appelle Biscuit."
    run_turn(store, brain, "Je travaille chez Safran.")
    work = run_turn(store, brain, "Où est-ce que je travaille ?")
    assert work.reply == "Tu travailles chez Safran."


def test_constraint_warning_fires_on_the_action_not_the_rule(tmp_path):
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    rule = run_turn(
        store,
        brain,
        "Do not restart the training gateway while pretraining is active.",
    )
    assert rule.memory_ok
    assert "about to act against" not in rule.reply
    assert any(fact.type == "constraint" for fact in store.active())
    action = run_turn(store, brain, "I'm about to restart the training gateway.")
    assert "about to act against" in action.reply
    assert "training gateway" in action.reply
    french = MemoryStore(tmp_path / "fr")
    stated = run_turn(
        french,
        brain,
        "Ne redémarre pas la passerelle d'entraînement pendant le pré-entraînement.",
    )
    assert "aller contre" not in stated.reply
    conflict = run_turn(french, brain, "Je vais redémarrer la passerelle.")
    assert "aller contre" in conflict.reply


def test_privacy_line_is_not_stored(tmp_path):
    store = MemoryStore(tmp_path)
    result = run_turn(store, ScriptedBrain(), "Forget this: my dog is called Pixel.")
    assert result.memory_ok
    assert result.receipt.stored == 0
    assert store.slot("fact:dog-name") is None


def test_more_french_slots_and_a_greeting_check_in(tmp_path):
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    named = run_turn(store, brain, "Je m'appelle Thomas.")
    assert named.memory_ok
    assert store.slot("fact:user-name").value == "Thomas"
    assert run_turn(store, brain, "Comment je m'appelle ?").reply == "Tu t'appelles Thomas."
    allergy = run_turn(store, brain, "Je suis allergique aux arachides.")
    assert allergy.memory_ok
    assert "arachides" in store.slot("health:allergy").value
    assert "arachides" in run_turn(store, brain, "À quoi suis-je allergique ?").reply
    home = run_turn(store, brain, "J'habite à Lyon.")
    assert store.slot("fact:home-city").value == "Lyon"
    assert run_turn(store, brain, "Où est-ce que j'habite ?").reply == "Tu habites à Lyon."
    appointment = run_turn(store, brain, "J'ai rendez-vous chez le médecin jeudi.")
    assert store.threads["thread:appointment"].status == "open"
    note = extract("N'oublie pas que le vélo est au garage.")
    assert note.facts[0].key.startswith("fact:note-")
    assert "vélo" in note.facts[0].value or "velo" in note.facts[0].value
    english = MemoryStore(tmp_path / "en")
    run_turn(english, brain, "My name is Thomas.")
    run_turn(english, brain, "I'm allergic to peanuts.")
    run_turn(english, brain, "I live in Lyon.")
    run_turn(english, brain, "I have an appointment with the doctor on Thursday.")
    assert english.slot("fact:user-name").value == "Thomas"
    assert english.slot("health:allergy").value == "peanuts"
    assert english.slot("fact:home-city").value == "Lyon"
    assert english.threads["thread:appointment"].priority == "high"
    quiet = run_turn(english, brain, "Hello")
    assert "check in" in quiet.reply
    assert "appointment" in quiet.reply.lower() or "doctor" in quiet.reply.lower()


def test_negation_does_not_store_the_denied_fact_and_correction_supersedes(tmp_path):
    denied = extract("Mon chien ne s'appelle pas Pixel.")
    assert denied.facts == []
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    run_turn(store, brain, "Mon chien s'appelle Pixel.")
    kept = run_turn(store, brain, "Mon chien ne s'appelle pas Pixel.")
    assert kept.receipt.stored == 0
    assert store.slot("fact:dog-name").value == "Pixel"
    renamed = run_turn(store, brain, "Non, en fait il s'appelle Biscuit.")
    assert renamed.memory_ok
    assert store.slot("fact:dog-name").value == "Biscuit"
    assert "remplace" in renamed.reply
    run_turn(store, brain, "Je travaille chez Airbus.")
    corrected = run_turn(store, brain, "Non, en fait je travaille chez Safran.")
    assert store.slot("job:employer").value == "Safran"
    assert corrected.receipt.stored == 1
    stopped = run_turn(store, brain, "Je ne travaille plus chez Safran.")
    assert stopped.memory_ok
    assert store.slot("job:employer").value == "none"
    dislike = extract("Je n'aime pas le lait de vache.")
    assert dislike.facts[0].key.startswith("preference:avoid-")
    assert "lait de vache" in dislike.facts[0].value
    assert "does not want" in dislike.facts[0].summary
    assert extract("Je n'ai pas de rendez-vous jeudi.").facts == []


def test_appointment_dates_resolve_demain_and_jeudi_prochain():
    wednesday = date(2026, 11, 11)
    assert resolve_fr_when("J'ai rendez-vous demain", wednesday) == date(2026, 11, 12)
    assert resolve_fr_when("rendez-vous jeudi prochain", wednesday) == date(2026, 11, 12)
    thursday = date(2026, 11, 12)
    assert resolve_fr_when("jeudi prochain", thursday) == date(2026, 11, 19)
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    soon = extract("J'ai rendez-vous demain chez le médecin.")
    assert soon.facts[0].key == "fact:appointment"
    assert tomorrow in soon.facts[0].value
    assert tomorrow in soon.facts[0].summary
    later = extract("J'ai rendez-vous jeudi prochain.")
    assert resolve_fr_when("jeudi prochain").isoformat() in later.facts[0].value


def test_done_with_closes_the_health_thread(tmp_path):
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    run_turn(store, brain, "My mum was diagnosed with breast cancer last week.")
    assert store.threads["thread:health-mum"].status == "open"
    closed = run_turn(store, brain, "I'm done with mum's diagnosis.")
    assert closed.memory_ok
    assert store.threads["thread:health-mum"].status == "closed"
