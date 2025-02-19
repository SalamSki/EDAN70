import regex as re 
import requests
import pandas as pd

def scrape_links(link):
  res = requests.get(f"https://runeberg.org{link}", headers={
    "Accept": "text/plain",
    "Connection": "keep-alive", 
})
  if res.status_code == 200:
    find_text_with_table = re.findall(r"(?<=<!-- mode=normal -->).+(?=<!-- NEWIMAGE2 ..>)",res.text, flags=re.S)
    if len(find_text_with_table) > 0:
      with open(f"./dataset/clear_text/NF_E{i}.txt", "a", encoding='utf-8') as f:
        text = re.sub(r"<table.+<\/table>|<br>|<i>|<\/i>|<span.*?>|<\/span>", "", find_text_with_table[0], flags=re.S)
        f.write(text+"\n")

for i in range(1,5):
  df = pd.read_csv(f"./ScrapedLinks/NF_E{i}.txt", header=None, names=["links"])

  for link in df['links']:
    scrape_links(link)