import requests



headers = {"User-Agent": "Mozilla/5.0"}

def getTopHeadlines(tag):
    url = f"https://finance.yahoo.com/quote/{tag}/news"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        headlines = []
        headlines.append(title)
        return headlines
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return []

print(getTopHeadlines())

