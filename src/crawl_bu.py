# src/crawl_bu_grad_clean_final.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import fitz  # PyMuPDF

# ===== START URLs for all BU graduate schools =====
START_URLS = [
    # "https://www.bu.edu/grad/admission-funding/graduate-admission/",
    # "https://www.bu.edu/cfa/admissions/",
     "https://www.bu.edu/met/admissions/apply-now-graduate/",
    # "https://www.bu.edu/cds-faculty/programs-admissions/",
    # "https://www.bu.edu/eng/admissions/graduate/",
    # "https://www.bu.edu/questrom/graduate-programs/admissions/",
    # "https://www.bu.edu/ssw/admissions/",
    # "https://www.bu.edu/wheelock/admissions-financial-aid/graduate-admissions/",
    # "https://www.bu.edu/com/admissions/",
    # "https://www.bu.edu/cas/graduate-student-resources/",
    # "https://www.bumc.bu.edu/gms/admissions/",
    # "https://www.bu.edu/pardeeschool/academics/graduate/",
    # "https://www.bu.edu/sargent/admissions/",
    # "https://www.bu.edu/hospitality/academics/graduate-programs/",
    # "https://www.bu.edu/sth/admissions/"
]

# ===== Allowed prefixes & keywords =====
ALLOWED_PATH_PREFIXES = [
    "/grad/",
    "/cfa/",
    "/met/",
    "/cds-faculty/",
    "/eng/",
    "/questrom/",
    "/ssw/",
    "/wheelock/",
    "/com/",
    "/cas/",
    "/gms/",
    "/pardeeschool/",
    "/sargent/",
    "/hospitality/",
    "/sth/"
]

ALLOWED_KEYWORDS = [
    "grad", "graduate", "admission", "apply", "application",
    "program", "academics", "degree", "curriculum", "requirements",
    "certificate", "ms-", "ma-", "mba", "phd"
]

# ===== Explicit exclusions =====
EXCLUDED_PATH_PARTS = [
    "stories", "/news/", "/event", "/calendar"
]

EXCLUDED_UNDERGRAD = [
    "/bs-", "/bls-", "bachelors"
]

MAX_DEPTH = 3

visited = set()
page_count = 0

# ===== Output folders =====
os.makedirs("data/html", exist_ok=True)
os.makedirs("data/pdf", exist_ok=True)
os.makedirs("data/text", exist_ok=True)


# ===== Filtering =====
def is_valid_link(link, base_netloc):
    parsed = urlparse(link)

    # Same domain or relative link
    if parsed.netloc not in ("", base_netloc):
        return False

    # Skip email/fragments
    if link.startswith("mailto:") or "#" in link:
        return False

    # Skip undergrad keyword
    if "undergraduate" in parsed.path.lower():
        return False

    # Must start with allowed school/admissions prefix
    if not any(parsed.path.lower().startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        return False

    # Must contain at least one grad/admissions keyword
    if not any(kw in parsed.path.lower() for kw in ALLOWED_KEYWORDS):
        return False

    # Skip excluded content categories
    if any(ex in parsed.path.lower() for ex in EXCLUDED_PATH_PARTS):
        return False

    # Skip undergrad programs
    if any(ex in parsed.path.lower() for ex in EXCLUDED_UNDERGRAD):
        return False

    return True


# ===== Save PDFs =====
def download_pdf(url):
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        pdf_path = os.path.join("data/pdf", os.path.basename(urlparse(url).path))
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        print(f"[PDF] Downloaded: {pdf_path}")

        # Extract PDF text
        pdf_doc = fitz.open(pdf_path)
        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text()
        text_filename = pdf_path.replace("data/pdf", "data/text").replace(".pdf", ".txt")
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(pdf_text)
        print(f"[PDF->TXT] Extracted: {text_filename}")

    except Exception as e:
        print(f"[ERROR] Failed PDF {url}: {e}")


# ===== Save HTML/Text =====
def save_page_text(url, text):
    filename = os.path.join(
        "data/text",
        urlparse(url).path.strip("/").replace("/", "_") + ".txt"
    )
    if not filename.endswith(".txt"):
        filename += ".txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[TXT] Saved: {filename}")


def save_html_text(url, base_netloc, depth):
    global page_count
    try:
        page_count += 1
        print(f"[{page_count}] Depth {depth} → Visiting: {url}")

        res = requests.get(url, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # Save raw HTML
        html_file = os.path.join(
            "data/html", urlparse(url).path.strip("/").replace("/", "_") + ".html"
        )
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(res.text)

        # Save cleaned text
        text = soup.get_text(separator="\n").strip()
        save_page_text(url, text)

        # Crawl deeper if under depth limit
        if depth < MAX_DEPTH:
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if link.lower().endswith(".pdf") and is_valid_link(link, base_netloc):
                    download_pdf(link)
                elif is_valid_link(link, base_netloc) and link not in visited:
                    crawl(link, base_netloc, depth + 1)

    except Exception as e:
        print(f"[ERROR] Failed HTML {url}: {e}")


# ===== Crawl =====
def crawl(url, base_netloc, depth=0):
    if url in visited:
        return
    visited.add(url)
    if url.lower().endswith(".pdf"):
        download_pdf(url)
    else:
        save_html_text(url, base_netloc, depth)


# ===== Main =====
if __name__ == "__main__":
    for start_url in START_URLS:
        base_netloc = urlparse(start_url).netloc
        crawl(start_url, base_netloc, depth=0)

    print(f"\n✅ Crawl complete. Total pages visited: {page_count}")
