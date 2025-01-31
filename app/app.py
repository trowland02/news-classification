import streamlit as st
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from classificationLib import NewsClassifier
from sentimentLib import SentimentModel

# ----------------------------------------------------------------------------------------------------------------------

def get_news():
    sources = [
        "http://feeds.bbci.co.uk/news/rss.xml"
        # "https://news.sky.com",
        # "https://www.ft.com",
        # "https://www.economist.com"
    ]

    classification_model = NewsClassifier()
    sentiment_model = SentimentModel()

    news_articles = _list_news(sources)
    for article in news_articles:
        classification = _classify_news(f"{article['title']}: {article['description']}", classification_model)
        sentiment = _sentiment_class(f"{article['title']}: {article['description']}", sentiment_model)
        article["classification"] = classification
        article["sentiment"] = sentiment
    
    return news_articles

# ----------------------------------------------------------------------------------------------------------------------

def _classify_news(text_to_classify, model):

    try:
        classification = model.predict(text_to_classify)
    except requests.exceptions.RequestException as e:
        print(f"Error calling model service: {e}")
        classification = "Classification error"

    return classification

# ----------------------------------------------------------------------------------------------------------------------

def _sentiment_class(text_to_sentiment, model):

    try:
        sentiment = model.score_news_content(text_to_sentiment)
    except requests.exceptions.RequestException as e:
        print(f"Error calling model service: {e}")
        sentiment = "Classification error"

    return sentiment

# ----------------------------------------------------------------------------------------------------------------------

def _get_soup(url):
    response = requests.get(url)
    if url.endswith(".xml"):
        soup = BeautifulSoup(response.text, "xml")
    else:
        soup = BeautifulSoup(response.text, "html.parser")
    return soup

# ----------------------------------------------------------------------------------------------------------------------

def _list_news(sources):
    articles = []
    for url in sources:
        soup = _get_soup(url)
        items = soup.find_all('item')

        for item in items:
            # pub_date = datetime.strptime(item.pubDate.text.strip(), '%Y-%m-%d %H:%M:%S') if item.pubDate else None
            pub_date = item.pubDate.text.strip()
            image_url = None

            # Try to extract the image URL from <media:thumbnail>
            thumbnail_tag = item.find('media:thumbnail')
            if thumbnail_tag and thumbnail_tag.get('url'):
                image_url = thumbnail_tag['url']

            article = {
                "title": item.title.text.strip().replace('\n', ' ') if item.title else None,
                "description": item.description.text.strip().replace('\n', ' ') if item.description else None,
                "link": item.link.text.strip().replace('\n', ' ') if item.link else None,
                "classification": "",  # Placeholder, will be updated later
                "sentiment": "",       # Placeholder, will be updated later
                "image_url": image_url,  # Image URL extracted
                "pubDate": pub_date
            }
            articles.append(article)
    return articles

# ----------------------------------------------------------------------------------------------------------------------

# Streamlit UI
def main():
    # Set up the Streamlit page
    st.title("News Article Classifier")
    st.write("Displaying the latest news articles with their classification and sentiment.")

    # Fetch the news articles
    news_articles = get_news()

    # Display news articles
    for article in news_articles:
        # Display article title and date
        st.header(article["title"])
        st.write(f"Published on: {article['pubDate']}")
        
        # Display the article's description
        st.write(article["description"])

        # Display article's classification
        st.write(f"**Classification:** {article['classification']}")

        # Display article's sentiment
        st.write(f"**Sentiment:** {article['sentiment']}")

        # Optionally display the image if available
        if article["image_url"]:
            st.image(article["image_url"], width=100)

        # Provide a link to the full article
        st.markdown(f"[Read full article]({article['link']})")
        
        # Add a horizontal line between articles for better readability
        st.markdown("---")

if __name__ == "__main__":
    main()