from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_rag_agent_path() -> None:
    """Add rag_agent parent to sys.path before any imports that need it (e.g. csv_ingestion)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--rag-agent-dir" and i + 1 < len(sys.argv):
            rag_agent_dir = Path(sys.argv[i + 1]).resolve()
            project_root = rag_agent_dir.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            return
        if arg.startswith("--rag-agent-dir="):
            val = arg.split("=", 1)[1]
            if val:
                rag_agent_dir = Path(val).resolve()
                project_root = rag_agent_dir.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
            return


_setup_rag_agent_path()

from preload.config import PreloadConfig
from preload.pipeline.backup import backup_persist_dir
from preload.pipeline.lock import FileLock
from preload.pipeline.report import RunReport
from preload.utils.logging import setup_logger
from preload.utils.paths import add_project_root_to_syspath

from preload.adapters.csv_adapter import CSVAdapter
from preload.adapters.web_adapter import WebPageListAdapter
from preload.adapters.pdf_adapter import PDFDirAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Manifest-driven preload pipeline (reuses rag_agent ingestion + chunking).")
    p.add_argument("--manifest", required=True, help="Path to manifest.yaml")
    p.add_argument("--persist-dir", required=True, help="Chroma persistence directory")
    p.add_argument("--collection", required=True, help="Chroma collection name")
    p.add_argument("--rag-agent-dir", required=True, help="Path to rag_agent directory (sibling to preload_pipeline)")
    p.add_argument("--backups-root", default=None, help="Backup root dir (default: <persist_parent>/backups)")
    p.add_argument("--keep-last", type=int, default=20, help="Keep newest N backups")
    p.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5", help="Embedding model (match rag_agent)")
    p.add_argument("--device", default="None", help="Device for embedding model (match rag_agent)")
    p.add_argument("--dry-run", action="store_true", help="Do everything except writing to Chroma")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    manifest_path = Path(args.manifest).resolve()
    rag_agent_dir = Path(args.rag_agent_dir).resolve()
    persist_dir = Path(args.persist_dir).resolve()
    backups_root = Path(args.backups_root).resolve() if args.backups_root else persist_dir.parent / "backups"

    # Ensure `import rag_agent...` works by adding the *parent* of rag_agent to sys.path
    add_project_root_to_syspath(rag_agent_dir)

    # Import after path is set up (rag_agent_integration imports rag_agent)
    from preload.rag_agent_integration import create_rag_agent_collection_and_utils

    cfg = PreloadConfig.from_manifest(
        manifest_path=manifest_path,
        persist_dir=persist_dir,
        collection_name=args.collection,
        backups_root=backups_root,
        keep_last=args.keep_last,
        embed_model=args.embed_model,
        device=args.device,
        dry_run=args.dry_run,
    )

    report = RunReport(manifest_path=str(manifest_path), persist_dir=str(persist_dir), collection=cfg.collection_name)

    lock_path = persist_dir.parent / ".preload.lock"
    with FileLock(lock_path, logger=logger):
        # Stage 0: backup (runs first)
        if persist_dir.exists():
            backup_path = backup_persist_dir(
                persist_dir=persist_dir,
                backups_root=cfg.backups_root,
                label="before_preload",
                keep_last=cfg.keep_last,
                logger=logger,
            )
            report.backup_path = str(backup_path)
        else:
            logger.info(f"Persist dir not found yet, creating new: {persist_dir}")
            persist_dir.mkdir(parents=True, exist_ok=True)

        # Create collection + ContentUtils using rag_agent classes
        collection, content_utils, web_adder, pdf_adder = create_rag_agent_collection_and_utils(
            persist_dir=cfg.persist_dir,
            collection_name=cfg.collection_name,
            embed_model=cfg.embed_model,
            device=cfg.device,
            dry_run=cfg.dry_run,
            logger=logger,
        )

        # Build adapters
        adapters = []
        for s in cfg.sources:
            st = s["type"].strip().lower()
            if st == "csv":
                adapters.append(CSVAdapter(s, collection=collection, content_utils=content_utils, dry_run=cfg.dry_run))
            elif st == "web_page_list":
                adapters.append(WebPageListAdapter(s, web_adder=web_adder, dry_run=cfg.dry_run))
            elif st == "pdf_dir":
                adapters.append(PDFDirAdapter(s, pdf_adder=pdf_adder, dry_run=cfg.dry_run))
            else:
                raise ValueError(f"Unknown source type: {st} (source={s.get('name')})")

        # Run
        for adapter in adapters:
            logger.info(f"Source: {adapter.source_name} ({adapter.source_type})")
            report.sources_started += 1
            try:
                stats = adapter.run(logger=logger)
                report.sources_succeeded += 1

                report.items_processed += stats.get("items_processed", 0)
                report.items_added += stats.get("items_added", 0)
                report.items_skipped += stats.get("items_skipped", 0)
                report.items_failed += stats.get("items_failed", 0)

            except Exception as e:
                report.sources_failed += 1
                report.errors.append({"source": adapter.source_name, "error": repr(e)})
                logger.exception(f"Failed source {adapter.source_name}: {e}")

    out_path = report.write_json(out_dir=persist_dir.parent, logger=logger)
    logger.info(f"Wrote report: {out_path}")
    logger.info(report.summary_str())
    return 0 if report.sources_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())