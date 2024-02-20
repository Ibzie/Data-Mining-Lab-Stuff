import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import pandas as pd
import time

root = "https://www.google.com/"
search_query = "Pakistan Travel"
results_limit = 1500

def fetch_news(link, count=0, results=[]):
    if count >= results_limit:
        print("Collected 1500 news items. Stopping...")
        return pd.DataFrame(results)

    try:
        response = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'})

        if response.status_code == 429:
            print("Rate limit hit. Waiting for 5 seconds.")
            time.sleep(5)
            return fetch_news(link, count, results)

        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.find_all('div', attrs={'class': 'Gx5Zad fP1Qef xpd EtOod pkphOe'}):
            if count >= results_limit:
                print("Collected 1500 news items. Stopping...")
                return pd.DataFrame(results)

            raw_link = item.find('a', href=True)['href']
            news_link = raw_link.split("/url?q=")[1].split('&sa=U&')[0]

            title = item.find('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}).get_text()
            full_description = item.find('div', attrs={'class': 'BNeawe s3v9rd AP7Wnd'}).get_text()

            if '·' in full_description:
                description, time_ago = full_description.rsplit(' · ', 1)
            else:
                description = full_description
                time_ago = "Time not specified"

            news_article = {
                'title': title,
                'description': description,
                'time_ago': time_ago,
                'link': news_link
            }

            print(f"Title: {title}")
            print(f"Link: {news_link}")
            print()

            results.append(news_article)
            count += 1
            time.sleep(1)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)
        return fetch_news(link, count, results)

    next_page = soup.find('a', attrs={'aria-label': 'Next page'})
    if next_page and count < results_limit:
        next_href = next_page['href']
        next_link = root + next_href
        return fetch_news(next_link, count, results)
    else:
        print("No more pages or reached 1500 news items.")
        return pd.DataFrame(results)

encoded_query = quote(search_query)
search_link = root + f"search?q={encoded_query}"

df = fetch_news(search_link)
print(df)
