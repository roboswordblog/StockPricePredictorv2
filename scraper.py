import xml.etree.ElementTree as ET
import requests


url = f"https://finance.yahoo.com/"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)

