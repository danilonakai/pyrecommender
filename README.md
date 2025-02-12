# Python-Based Statistical NLP Article Recommender

This is a simple recommender system for text documents built using Python. It loads a list of articles and recommends similar ones based on a user's reading history. The system uses cosine similarity to calculate article relevance and provides both highly similar and diverse recommendations to help users explore content beyond their immediate preferences.

## Features
- **Random Article Selection**: Initially, a random article is displayed to the user.
- **Article Recommendations**: After reading an article, the system generates a list of recommended articles based on similarity to the last read article.
- **Similarity Calculation**: Utilizes cosine similarity between document vectors to find similar articles.
- **Diverse Recommendations**: In addition to the most similar articles, a few less similar articles are recommended to encourage exploration.
- **Customizable Parameters**: The system supports configuration of the number of recommendations, minimum and maximum document frequency, and n-grams for better vectorization.

## Dependencies
- **Python**
- **scikit-learn**: For vectorization and similarity calculation
- **json**, **csv**: For handling article data

## Installation
1. Clone the repository to your local machine.
2. Open the project in PyCharm (Community Edition) or any Python IDE of your choice.
3. Ensure you are using Python 3.7+ for the best compatibility with the dependencies.
5. Install dependencies
   ```sh
   pip install sklearn
   ```

## How It Works
- **Load Articles**: Articles are loaded from a CSV or JSON file containing fields such as `title` and `text`.
- **Vectorization**: The text of each article is vectorized using `TfidfVectorizer` from `scikit-learn`. This transforms the raw text into numerical vectors suitable for similarity calculations.
- **Cosine Similarity**: The system calculates the cosine similarity between the vector of the last article the user read and all other articles in the dataset.
- **Recommendations**: Based on the similarity scores, a set of similar articles is recommended. To add diversity, a couple of less similar articles are included.
- **User Interaction**: The user can choose an article from the recommendations, after which the process repeats, showing more relevant articles based on their choice.

## Running the Project
To run the recommender system:

1. Prepare a json or CSV file with your articles. (Change filename in the project if necessary) OR use the sample data provided to test
2. Execute the script OR run the recommender.py file in PyCharm (Community Edition) or any Python IDE of your choice:
    ```bash
    python recommender.py
    ```
3. Follow the prompts to interact with the recommender system.

## Sample Output
![image](https://github.com/user-attachments/assets/803b3b1b-bdbc-4040-9129-d6399d0758a7)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
