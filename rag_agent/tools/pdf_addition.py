from typing import Dict, List, Any, Optional
import pdfplumber
import re
from rag_agent.utils.metadata import build_canonical_chunk_metadata


class PDFAddition:
    """
    Service responsible for ingesting page-aware PDF content
    into a vector database with deduplication and chunking.
    """

    def __init__(
        self,
        collection,
        content_utils,
        null_str: str = "",
    ):
        """
        Args:
            collection: Vector database collection instance
            content_utils: Instance of ContentUtils
            null_str: Placeholder value for empty metadata fields
        """
        self.collection = collection
        self.content_utils = content_utils
        self.null_str = null_str

    def clean_pdf_text(self, text: str) -> str:
        # Remove hyphenation at line breaks: "exam-\nple" → "example"
        text = re.sub(r"-\n(\w+)", r"\1", text)

        # Replace line breaks within paragraphs with spaces
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Normalize multiple newlines
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def _table_to_row_narratives(self, table: List[List[Any]]) -> str:
        """Convert a table (list of rows) to row-as-narrative format.

        Each data row becomes "header1 is value1 header2 is value2 ...".
        First row is treated as headers; subsequent rows as data.

        Args:
            table: List of rows, each row is a list of cell values

        Returns:
            Newline-separated narrative paragraphs, one per data row
        """
        if not table or len(table) < 2:
            return ""

        headers = []
        for cell in table[0]:
            val = (cell or "").strip() if cell is not None else ""
            headers.append(val if val else f"Column{len(headers)}")

        rows_text = []
        for row in table[1:]:
            parts = []
            for i, cell in enumerate(row):
                header = headers[i] if i < len(headers) else f"Column{i}"
                val = (cell or "").strip() if cell is not None else ""
                parts.append(f"{header} is {val}")
            rows_text.append(" ".join(parts).strip())

        return "\n".join(r for r in rows_text if r)

    def extract_pdf_pages(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extracts and cleans text from each page of a PDF.

        Uses pdfplumber for extraction. Body text and tables are combined;
        each table row is converted to narrative format ("header is value ...").

        Args:
            pdf_path: Local filesystem path to the PDF file

        Returns:
            List of dicts with page number and cleaned text:
            [{"page": int, "text": str}]
        """
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""

                tables = page.extract_tables() or []
                table_texts = []
                for table in tables:
                    narrative = self._table_to_row_narratives(table)
                    if narrative:
                        table_texts.append(narrative)

                combined = raw_text
                if table_texts:
                    if combined:
                        combined += "\n\n"
                    combined += "\n\n".join(table_texts)

                if not combined.strip():
                    continue

                clean_text = self.clean_pdf_text(combined)
                if clean_text:
                    pages.append({
                        "page": page_num,
                        "text": clean_text
                    })

        return pages


    def add_pdf_content(
        self,
        *,
        pdf_path: str,
        source_id: str,
        title: str,
        location: Optional[str] = None,
        month_year: Optional[str] = None,
        language: str = "en"
    ) -> Dict:
        """
        Adds page-aware PDF content to the vector database.

        Use this tool when:
        - The user provides a PDF
        - New documents need to be indexed for retrieval

        Guarantees:
        - Token-based chunking
        - Hash-based deduplication
        - Idempotent ingestion

        Args:
            pdf_path: Local filesystem path to the PDF file
            source_id: Unique identifier for the PDF source
            title: Human-readable document title
            location: Optional geographic context
            month_year: Optional publication date
            language: Language code (default: "en")

        Returns:
            Success:
            {
                "status": "success",
                "source_type": "pdf",
                "source_id": str,
                "chunks_added": int,
                "chunks_skipped_as_duplicates": int
            }

            Error:
            {
                "status": "error",
                "error_message": str
            }
        """
        location = location.upper() if location else self.null_str
        month_year = month_year.strip() if month_year else self.null_str

        try:
            pdf_pages = self.extract_pdf_pages(pdf_path)
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Failed to extract PDF text: {str(e)}",
            }

        if not pdf_pages:
            return {
                "status": "error",
                "error_message": "No extractable text found in PDF",
            }

        documents = []
        metadatas = []
        ids = []

        added = 0
        skipped = 0
        seen_hashes = set()

        for page_data in pdf_pages:
            page_num = page_data["page"]
            page_text = page_data["text"]

            chunks = self.content_utils.chunk_by_tokens(
                page_text,
                max_tokens=self.content_utils.chunk_config["pdf"]["max_tokens"],
                overlap=self.content_utils.chunk_config["pdf"]["overlap"],
            )

            for chunk_index, chunk in enumerate(chunks):
                content_hash = self.content_utils.compute_content_hash(chunk)

                if content_hash in seen_hashes:
                    skipped += 1
                    continue

                if self.content_utils.content_hash_exists(
                    self.collection,
                    content_hash
                ):
                    skipped += 1
                    continue

                seen_hashes.add(content_hash)
                added += 1

                doc_id = f"{source_id}_p{page_num}_c{chunk_index}"

                documents.append(f"Title: {title}\n\n{chunk}")
                metadatas.append(
                    build_canonical_chunk_metadata(
                        source_type="pdf",
                        source_id=source_id,
                        title=title,
                        url=self.null_str,
                        page=page_num,
                        chunk_index=chunk_index,
                        location=location,
                        month_year=month_year,
                        content_hash=content_hash,
                        language=language,
                    )
                )
                ids.append(doc_id)

        if not documents:
            return {
                "status": "error",
                "error_message": "No new content to add after deduplication",
            }

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        return {
            "status": "success",
            "source_type": "pdf",
            "source_id": source_id,
            "chunks_added": added,
            "chunks_skipped_as_duplicates": skipped,
        }