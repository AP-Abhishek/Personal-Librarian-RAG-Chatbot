import sys, shutil, logging, warnings, os, gc, atexit, signal, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

from src.retrieval.retriever import load_user_vectorstore, get_retriever
from src.generation.llm import load_llm
from src.generation.rag_chain import run_rag
from src.memory.conversation_memory import ConversationMemory
from src.embeddings.build_vectorstore import build_user_vectorstore, delete_pdf_from_user_vectorstore
from src.utils import format_chat_export_text, format_chat_export_markdown

USER_ID = "cli_user"
UPLOAD_DIR = Path(f"data/uploads/{USER_ID}/pdfs")
VECTORSTORE_PATH = Path(f"db/chroma/{USER_ID}")
MEMORY_PATH = Path(f"data/memory/{USER_ID}/conversation.json")

def cleanup_cli_session():
    gc.collect()
    if VECTORSTORE_PATH.exists():
        shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR.parent, ignore_errors=True)
    if MEMORY_PATH.exists():
        shutil.rmtree(MEMORY_PATH.parent, ignore_errors=True)

atexit.register(cleanup_cli_session)

def _signal_handler(sig, frame):
    cleanup_cli_session()
    sys.exit(0)

try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)
except Exception:
    pass

def ensure_vectorstore() -> bool:
    if VECTORSTORE_PATH.exists() and list(VECTORSTORE_PATH.glob("*")):
        return True
    
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
    if pdf_files:
        print(f"📄  Found {len(pdf_files)} PDF(s) in {UPLOAD_DIR}. Building library...")
        try:
            build_user_vectorstore(USER_ID)
            print("✅  Library built successfully!\n")
            return True
        except Exception as e:
            print(f"❌  Error building vectorstore: {e}")
            return False
    else:
        print("⚠️  No indexed library found.")
        print(f"👉  Upload a PDF with `/upload <filepath>` or place files in: {UPLOAD_DIR.resolve()}\n")
        return False

def main():
    print("\n==========================================")
    print("📚  Personal Librarian RAG Chatbot (CLI)")
    print("==========================================")
    print("Commands:")
    print("  /upload <path>   : Copy & index a PDF file (e.g. /upload \"C:\\doc.pdf\")")
    print("  /list            : List all indexed PDFs in the library")
    print("  /delete <fname>  : Delete a PDF file by name or path from the library")
    print("  /clean           : Wipe all uploaded PDFs, vectorstore & conversation data")
    print("  /logs            : Print recorded query log audit transcript")
    print("  /verbose         : Toggle verbose RAG logs on/off for future queries")
    print("  /clear           : Clear conversation chat history & memory")
    print("  /export          : Save chat transcript to cli_chat_history.txt")
    print("  /exit            : Quit the chatbot & clean up session files\n")

    has_library = ensure_vectorstore()
    
    llm = load_llm()
    memory = ConversationMemory(max_size=5, persist_path=MEMORY_PATH)
    chat_history = []
    log_audit_records = []
    verbose_logs = False
    
    retriever = None
    if has_library:
        try:
            vectorstore = load_user_vectorstore(USER_ID)
            retriever = get_retriever(vectorstore)
        except Exception as e:
            print(f"Warning loading vectorstore: {e}")

    try:
        while True:
            try:
                user_query = input("User > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session...")
                break

            if not user_query:
                continue

            cmd = user_query.lower()

            if cmd in ["/exit", "exit", "/quit", "quit"]:
                print("Goodbye!")
                break

            if cmd in ["/logs", "logs"]:
                if not log_audit_records:
                    print("📊  No execution logs recorded yet.\n")
                else:
                    print("📊  === Execution Log Audit Transcript ===")
                    for idx, record in enumerate(log_audit_records, 1):
                        print(f"   {idx}. [{record['timestamp']}] Query: \"{record['query']}\"")
                        print(f"      Result: {record['result']} | Confidence: {record['confidence']} | Sources: {record['sources_count']}")
                    print()
                continue

            if cmd in ["/verbose", "verbose"]:
                verbose_logs = not verbose_logs
                new_level = logging.INFO if verbose_logs else logging.WARNING
                logging.getLogger().setLevel(new_level)
                logging.getLogger("src.generation.rag_chain").setLevel(new_level)
                status = "ENABLED" if verbose_logs else "DISABLED"
                print(f"📊  Verbose RAG execution logs are now {status}.\n")
                continue

            if cmd.startswith("/upload") or cmd.startswith("upload "):
                raw_arg = user_query.split(maxsplit=1)
                if len(raw_arg) < 2:
                    print("⚠️  Usage: /upload <filepath> (e.g. /upload \"C:\\path\\to\\document.pdf\")\n")
                    continue

                src_path_str = raw_arg[1].strip('"\'')
                src_path = Path(src_path_str)

                if not src_path.exists() or not src_path.is_file():
                    print(f"❌  Error: File not found at '{src_path_str}'\n")
                    continue

                if src_path.suffix.lower() != ".pdf":
                    print(f"❌  Error: '{src_path.name}' is not a PDF file.\n")
                    continue

                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                dest_path = UPLOAD_DIR / src_path.name
                shutil.copy2(src_path, dest_path)
                print(f"📄  Copied '{src_path.name}' to library. Building vector embeddings...")

                try:
                    build_user_vectorstore(USER_ID)
                    vectorstore = load_user_vectorstore(USER_ID)
                    retriever = get_retriever(vectorstore)
                    print(f"✅  Successfully indexed '{src_path.name}'! Library updated.\n")
                except Exception as e:
                    print(f"❌  Error indexing PDF: {e}\n")
                continue

            if cmd in ["/list", "list"]:
                pdfs = list(UPLOAD_DIR.glob("*.pdf")) if UPLOAD_DIR.exists() else []
                if not pdfs:
                    print("📚  Library is currently empty.\n")
                else:
                    print(f"📚  Indexed Documents ({len(pdfs)}):")
                    for idx, pdf in enumerate(pdfs, 1):
                        size_kb = pdf.stat().st_size / 1024
                        print(f"   {idx}. 📄  {pdf.name} ({size_kb:.1f} KB)")
                    print()
                continue

            if cmd.startswith("/delete") or cmd.startswith("delete "):
                prefix_len = 7 if cmd.startswith("/delete") else 6
                raw_arg = user_query[prefix_len:].strip().strip('"\'')
                if not raw_arg:
                    print("⚠️  Usage: /delete <filename or path> (e.g. /delete \"document.pdf\")\n")
                    continue

                pdf_name = Path(raw_arg).name
                target_pdf = UPLOAD_DIR / pdf_name
                
                if not target_pdf.exists():
                    existing_map = {p.name.lower(): p for p in UPLOAD_DIR.glob("*.pdf")} if UPLOAD_DIR.exists() else {}
                    if pdf_name.lower() in existing_map:
                        target_pdf = existing_map[pdf_name.lower()]
                        pdf_name = target_pdf.name
                    else:
                        print(f"❌  Error: File '{pdf_name}' not found in library. Type '/list' to see files.\n")
                        continue

                delete_pdf_from_user_vectorstore(USER_ID, pdf_name)
                target_pdf.unlink(missing_ok=True)

                remaining = list(UPLOAD_DIR.glob("*.pdf"))
                if remaining:
                    vectorstore = load_user_vectorstore(USER_ID)
                    retriever = get_retriever(vectorstore)
                    print(f"🗑️  Deleted '{pdf_name}'. Library updated ({len(remaining)} PDFs remaining).\n")
                else:
                    retriever = None
                    gc.collect()
                    if VECTORSTORE_PATH.exists():
                        shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
                    print(f"🗑️  Deleted '{pdf_name}'. Library is now empty.\n")
                continue

            if cmd in ["/clean", "clean"]:
                chat_history.clear()
                memory.clear()
                log_audit_records.clear()
                retriever = None
                gc.collect()
                if VECTORSTORE_PATH.exists():
                    shutil.rmtree(VECTORSTORE_PATH, ignore_errors=True)
                if UPLOAD_DIR.exists():
                    shutil.rmtree(UPLOAD_DIR.parent, ignore_errors=True)
                if MEMORY_PATH.exists():
                    shutil.rmtree(MEMORY_PATH.parent, ignore_errors=True)
                print("🧹  Wiped all CLI session data, PDF documents, and vectorstore library!\n")
                continue

            if cmd in ["/clear", "clear"]:
                chat_history.clear()
                memory.clear()
                print("🧹  Chat history and memory cleared.\n")
                continue

            if cmd in ["/export", "export"]:
                if not chat_history:
                    print("⚠️  No chat history to export.\n")
                    continue
                txt_export = format_chat_export_text(chat_history)
                out_file = Path("cli_chat_history.txt")
                out_file.write_text(txt_export, encoding="utf-8")
                print(f"📤  Chat transcript exported to: {out_file.resolve()}\n")
                continue

            if not retriever:
                print("⚠️  Library is empty. Upload a PDF first using: /upload <filepath>\n")
                continue

            if memory.is_vague(user_query):
                last_query = memory.get_last_meaningful_query()
                if not last_query:
                    clarification = "Could you please clarify what specific topic or document section you're referring to?"
                    print(f"Librarian > {clarification}\n")
                    chat_history.append({"role": "user", "content": user_query})
                    chat_history.append({"role": "assistant", "content": clarification, "sources": []})
                    memory.add_user_query(user_query)
                    continue
                final_query = f"{last_query}. {user_query}"
            else:
                final_query = user_query

            chat_history.append({"role": "user", "content": user_query})
            memory.add_user_query(user_query)

            print("Librarian > Thinking...")
            try:
                result = run_rag(llm, retriever, final_query)
                answer = result["answer"]
                sources = result.get("sources", [])
                conf = result.get("confidence", 0.0)
                status_str = "ANSWERED" if sources else "REFUSED"
            except Exception as e:
                answer = f"An error occurred while consulting the library: {e}"
                sources = []
                conf = 0.0
                status_str = "ERROR"

            log_audit_records.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "query": user_query,
                "result": status_str,
                "confidence": conf,
                "sources_count": len(sources)
            })

            print(f"\nLibrarian > {answer}")
            if sources:
                print("\n📌  Sources & References:")
                for src in sources:
                    if isinstance(src, dict):
                        print(f"   - 📄  {src.get('pdf', 'Doc')} (Page {src.get('page', '1')})")
                        print(f"     \"{src.get('snippet', '')[:120]}...\"")
                    else:
                        print(f"   - {src}")
            print("-" * 50 + "\n")

            chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
    finally:
        cleanup_cli_session()

if __name__ == "__main__":
    main()
