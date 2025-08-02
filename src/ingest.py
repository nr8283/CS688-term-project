import requests
from bs4 import BeautifulSoup
import os

URLS = [
    # "https://www.bu.edu/met/admissions/graduate-admissions/",
    # "https://www.bu.edu/met/admissions/application-deadlines/",
    "https://www.bu.edu/grad/"
]

os.makedirs("data", exist_ok=True)

for url in URLS:
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text(separator="\n")
    filename = os.path.join("data", url.split("/")[-2] + ".txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)