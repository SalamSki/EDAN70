import requests
import regex as re 

for i in range(9,1580):
  res = requests.get(f"https://runeberg.org/nfaa/{i:04}.html")
  html_converter = [("&", "&amp;"),("<", "&lt;"),(">", "&gt;"),('"', "&quot;"),("'", "&apos;"),("<>", "&lt;&gt;")]

  if res.status_code == 200:
    find_text_with_table = re.findall(r"(?<=<!-- mode=normal -->).+(?=<!-- NEWIMAGE2 ..>)",res.text, flags=re.S)
    if len(find_text_with_table) > 0:
      with open("edition1.txt", "a", encoding='utf-8') as f:
        find_text_with_table = find_text_with_table[0]
        text = re.sub("<table.+<\/table>|<br>|<i>|<\/i>|<span.*?>|<\/span>", "", find_text_with_table, flags=re.S)
        for wanted,found in html_converter:
          text = re.sub(found, wanted,text)
        print(f"{i:04}")
        f.write(text+"\n")
    else:
      print(f"--{i:04}")