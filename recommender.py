from csv import DictReader
from random import randint
import json

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_RECS = 10 # number of recommendations to return to the user
LESS_SIMILAR_ARTICLES = 2 # number of less similar articles

def load_articles(filename, num=None, filetype="csv"):
    articles = []
    if filetype=="csv":
        with open(filename, encoding="utf-8") as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                articles.append(row)
    elif filetype=="json":
        with open(filename, encoding="utf-8") as jsonfile:
            articles = json.loads(jsonfile.read())
    for row in articles:
        if row["title"] == None:
            row["title"] = row["text"][:30]
    if num:
        shuffle(articles)
        articles = articles[:num]
    print(len(articles),"articles loaded")
    return articles

def init_recommendations(n, articles):
    recommendations = []
    for _  in range(n):
        article = randint(0, len(articles)-1)
        while article in recommendations:
            article = randint(0, len(articles)-1)
        recommendations.append(article)
    return recommendations

def display_recommendations(recommendations, articles, loop_count):
    if loop_count > 0:
        loop_range = range(len(recommendations) - LESS_SIMILAR_ARTICLES)
    else:
        loop_range = range(len(recommendations))

    print("\n\n\nHere are some new recommendations for you:\n")
    for i in loop_range:
        art_num = recommendations[i]
        print(str(i+1)+".",articles[art_num]["title"])

    # Add less similar options if it's not the initial loop
    if loop_count > 0:
        print("\nOr if you want something different, how about...:\n")

        start_index = len(recommendations) - LESS_SIMILAR_ARTICLES

        for i in range(start_index, len(recommendations)):
            art_num = recommendations[i]
            print(str(i + 1) + ".", articles[art_num]["title"])

def display_article(art_num, articles):
    """Displays article 'art_num' from the articles"""
    print("\n\n")
    print("article",art_num)
    print("=========================================")
    print(articles[art_num]["title"])
    print()
    print(articles[art_num]["text"])
    print("=========================================")
    print("\n\n")

def new_recommendations(last_choice, n, article_vectors):
    """
    Generates article recommendations based on the cosine similarity between articles.

    This function suggests new articles to a user based on the similarity to the last article they read,
    prioritizing articles with high similarity and also including a few less similar ones.

    - The cosine similarity is used to measure the similarity between the article vectors.
    - The function selects the most similar articles excluding the `last_choice`, as the user has already read that one.
    - The function also selects a few less similar articles for diversity in the recommendations.

    Reference:
    https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    """

    last_vector = article_vectors[last_choice]

    # Compute similarity between the last article and all others
    similarities = cosine_similarity(last_vector, article_vectors)
    similarity_scores = similarities[0]

    # Sort by similarity
    similar_articles = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)

    similar_articles_count = n - LESS_SIMILAR_ARTICLES

    # Select recommendations excluding last choice
    recommendations = [idx for idx, score in similar_articles if idx != last_choice][:similar_articles_count]

    # Select the less similar articles
    less_similar_articles = [idx for idx, score in similar_articles if idx != last_choice][-LESS_SIMILAR_ARTICLES:]

    # Combine the most similar articles with the less similar articles
    recommendations.extend(less_similar_articles)

    return recommendations

def vectorize_articles(articles):
    """
        Uses the TfidfVectorizer from sklearn to convert the articles into vectors.

        The vectorizer is configured to:
            - Ignore terms that appear in more than 80% of the documents (`max_df=0.8`).
            - Include terms that appear in at least 2 documents (`min_df=2`).
            - Remove common English stop words using the 'english' stop words setting.

        Reference:
        https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    texts = [article["text"] for article in articles]
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
    article_vectors = vectorizer.fit_transform(texts)
    return article_vectors

def main():
    articles = load_articles('data/wikipedia_sample.json',filetype="json")
    article_vectors = vectorize_articles(articles)

    print("\n\n")
    recs = init_recommendations(NUM_RECS, articles)
    loop_count = 0

    while True:
        display_recommendations(recs, articles, loop_count)
        choice = int(input("\nYour choice? "))-1
        if choice < 0 or choice >= len(recs):
            print("Invalid Choice. Goodbye!")
            break
        display_article(recs[choice], articles)
        input("Press Enter")
        recs = new_recommendations(recs[choice], NUM_RECS, article_vectors)
        loop_count += 1

if __name__ == "__main__":
    main()