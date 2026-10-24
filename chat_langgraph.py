import asyncio

from ingestion import ingest_document
from pdf_export import export_to_pdf
from Team_graph import ask, resume


def prompt_for_decision(pending: dict) -> dict:
    print("\n--- APPROVAL NEEDED: this email has NOT been sent ---")
    print("To:     ", ", ".join(pending["to"]))
    print("Subject:", pending["subject"])
    print("Body:\n" + pending["body"])
    print("-----------------------------------------------------")

    while True:
        choice = input("approve / edit / reject (a/e/r): ").strip().lower()
        if choice in ("a", "approve"):
            return {"action": "approve"}
        if choice in ("r", "reject"):
            reason = input("reason (optional): ").strip() or "rejected by reviewer"
            return {"action": "reject", "reason": reason}
        if choice in ("e", "edit"):
            decision = {"action": "edit"}
            to = input("new recipients, comma separated (blank = keep): ").strip()
            subject = input("new subject (blank = keep): ").strip()
            body = input("new body (blank = keep): ").strip()
            if to:
                decision["to"] = [address.strip() for address in to.split(",") if address.strip()]
            if subject:
                decision["subject"] = subject
            if body:
                decision["body"] = body
            return decision
        print("Please type a, e or r.")


async def main():
    print("Ask away (type exit to leave)")
    print("  /file <path>  attach a document to your next question")
    print("  /pdf <path>   save the last answer as a PDF")

    document_text = None
    last_answer = None
    thread_id = None

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        if user_input.startswith("/file "):
            try:
                document_text = ingest_document(user_input[6:].strip().strip('"'))
                print("Document attached to your next question.")
            except Exception as exc:
                print(f"Could not read that file: {exc}")
            continue

        if user_input.startswith("/pdf "):
            if last_answer:
                try:
                    path = export_to_pdf(
                        last_answer,
                        user_input[5:].strip().strip('"'),
                        title="Answer",
                    )
                    print("Saved:", path)
                except Exception as exc:
                    print(f"Could not create PDF: {exc}")
            else:
                print("No answer to export yet.")
            continue

        try:
            result = await ask(user_input, document_text, thread_id=thread_id)
            document_text = None
            thread_id = result.get("thread_id", thread_id)

            while result["status"] == "pending_approval":
                decision = prompt_for_decision(result["approval"])
                result = await resume(result["thread_id"], decision)
                thread_id = result.get("thread_id", thread_id)

            last_answer = result.get("answer", "")
            print(f"Bot answer: {last_answer}")
        except Exception as exc:
            print(f"Something went wrong: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
