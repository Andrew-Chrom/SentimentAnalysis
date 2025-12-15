"""
Forming .json file from restaurants_with_ids.html which i got in get_restaurant_id_html.py
"""

from bs4 import BeautifulSoup


import json

filename = 'restaurants_with_ids.html'

with open(filename, 'r', encoding='utf-8') as file:
    data = file.read()

soup = BeautifulSoup(data, 'html.parser')


# we need data-id attribute for creating url request for the review in get_reviews.py
finded = soup.find_all('div', attrs={'data-id': True})

data_ids = []
for elem in finded:
    data_ids.append(elem['data-id'])


# saving .json file with ids
json_data = json.dumps(data_ids, indent=4, ensure_ascii=False)
with open('restaurant_ids.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

# print(finded)