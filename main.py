"""Command-line entry point of HRagentAssistant.

    python main.py init               create the database, demo accounts and import the demo corpus
    python main.py ask "question"     ask a question as a demo user (local developer convenience)
    python main.py documents          list the documents visible to a demo user
    python main.py eval               run the offline evaluation and print the report
    python main.py build-samples      regenerate the DOCX/PDF demo documents
    python main.py serve              start the Streamlit interface

The CLI works directly on the local database: ``--as USER`` selects a demo user *without* a password
(anyone who can run this already has file access to the data). The web interface always requires a login.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hr_assistant import auth
from hr_assistant.config import BASE_DIR, Settings
from hr_assistant.logger import configure_logging


def _reconfigure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _app(args):
    from hr_assistant.pipeline import build_app

    settings = Settings.from_env()
    configure_logging(settings.data_dir / "logs")
    return build_app(settings)


def cmd_init(args) -> int:
    from hr_assistant import demo

    app = _app(args)
    try:
        created = demo.seed_users(app)
        credentials_path = app.settings.data_dir / "demo_credentials.txt"
        if created:
            demo.write_credentials(credentials_path, created)
            print(f"Comptes de démonstration créés : {', '.join(created)}")
            print(f"Mots de passe écrits dans : {credentials_path} (fichier local, ignoré par git)")
        else:
            print("Comptes de démonstration déjà présents (mots de passe inchangés).")
        hr = demo.demo_hr_user(app)
        sets = [app.settings.samples_dir] + ([app.settings.special_samples_dir] if args.special else [])
        for directory in sets:
            report = demo.import_sample_dir(app, hr, directory)
            print(
                f"{directory.name}: {len(report.imported)} importé(s), {len(report.skipped)} déjà présent(s), "
                f"{len(report.failed)} en échec"
            )
            for name, reason in report.failed.items():
                print(f"  ! {name}: {reason}")
        return 0
    finally:
        app.close()


def _resolve_user(app, username: str):
    user = auth.get_user_by_username(app.db, username)
    if user is None:
        print(f"Utilisateur « {username} » introuvable. Lancez d'abord : python main.py init", file=sys.stderr)
        raise SystemExit(2)
    return user


def cmd_ask(args) -> int:
    app = _app(args)
    try:
        user = _resolve_user(app, args.user)
        reply = app.handle_message(user, args.question)
        if reply.answer is None:
            print(reply.message)
            return 0
        answer = reply.answer
        print(f"[{answer.status} · mode={answer.mode}]")
        print(answer.text)
        for citation in answer.citations:
            print(f"  {citation.marker} — {citation.label()}")
        for warning in answer.warnings:
            print(f"  ! {warning}")
        if answer.suggest_handoff:
            print("→ Un transfert vers un responsable RH est proposé.")
        return 0
    finally:
        app.close()


def cmd_documents(args) -> int:
    app = _app(args)
    try:
        user = _resolve_user(app, args.user)
        for doc in app.documents.list_documents(user):
            stale = " (à réindexer)" if doc.indexing_status == "indexed" and doc.collection != app.index.collection else ""
            print(
                f"{doc.name:42} {doc.doc_type:5} v{doc.version or '-':6} {doc.status:9} {doc.access_level:8} "
                f"{doc.indexing_status}{stale} ({doc.chunk_count} passages)"
            )
        return 0
    finally:
        app.close()


def cmd_eval(args) -> int:
    from hr_assistant.evaluation import DEFAULT_CASES_PATH, run_local_evaluation

    overrides = None
    if args.embeddings == "jina":  # evaluates the semantic backend (needs JINA_API_KEY; sends synthetic text to Jina)
        env = Settings.from_env()
        overrides = {"embedding_backend": "jina", "embedding_model": env.embedding_model, "jina_api_key": env.jina_api_key}
    report = run_local_evaluation(Path(args.cases) if args.cases else DEFAULT_CASES_PATH, overrides)
    print(report.to_markdown())
    return 0 if report.passed == report.total else 1


def cmd_build_samples(args) -> int:
    from hr_assistant.demo import build_binary_samples

    for path in build_binary_samples(Settings().samples_dir):
        print(f"écrit : {path.relative_to(BASE_DIR)}")
    return 0


def cmd_serve(args) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(BASE_DIR / "app.py")])


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdout()
    parser = argparse.ArgumentParser(prog="main.py", description="HRagentAssistant - prototype d'assistant RH documentaire")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="crée la base, les comptes de démonstration et importe le corpus")
    p_init.add_argument("--special", action="store_true", help="importe aussi data/samples_special (versions, conflits, document piégé)")
    p_init.set_defaults(func=cmd_init)

    p_ask = sub.add_parser("ask", help="pose une question")
    p_ask.add_argument("question")
    p_ask.add_argument("--as", dest="user", default="alice", help="utilisateur de démonstration (défaut : alice)")
    p_ask.set_defaults(func=cmd_ask)

    p_docs = sub.add_parser("documents", help="liste les documents visibles")
    p_docs.add_argument("--as", dest="user", default="claire")
    p_docs.set_defaults(func=cmd_documents)

    p_eval = sub.add_parser("eval", help="évaluation locale hors ligne")
    p_eval.add_argument("--cases", help="fichier JSON de cas (défaut : data/eval/hr_eval_cases.json)")
    p_eval.add_argument(
        "--embeddings", choices=["hashing", "jina"], default="hashing",
        help="backend d'embeddings évalué (jina : appels réseau, documents synthétiques uniquement)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_build = sub.add_parser("build-samples", help="régénère les documents DOCX/PDF de démonstration")
    p_build.set_defaults(func=cmd_build_samples)

    p_serve = sub.add_parser("serve", help="lance l'interface Streamlit")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
