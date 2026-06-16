"""
Predefined test scenarios for evaluating the chatbot recommender.

Each test case contains:
- user_input: Natural language description of movie preferences
- relevant_movie_ids: List of movieIds that should be recommended (ground truth)
- description: Explanation of the test case

movieIds are from the MovieLens / TMDB 5000 dataset (movies_final.csv).
"""

TEST_SCENARIOS = [
    {
        "id": "test_01_action_movie",
        "user_input": "I want an action movie with lots of fights and explosions",
        "relevant_movie_ids": [58559, 2019, 1196, 7153, 79132],
        "description": "Action movie with fighting and explosions",
    },
    {
        "id": "test_02_comedy",
        "user_input": "recommend me a funny comedy movie that will make me laugh",
        "relevant_movie_ids": [1870, 356, 3462, 1477, 909],
        "description": "Funny comedy for entertainment",
    },
    {
        "id": "test_03_drama_emotional",
        "user_input": "I'm looking for an emotional drama with touching storylines",
        "relevant_movie_ids": [1870, 318, 858, 527, 112552],
        "description": "Emotional drama film",
    },
    {
        "id": "test_04_scifi_future",
        "user_input": "Find me a sci-fi movie with robots and futuristic space themes",
        "relevant_movie_ids": [1196, 109487, 79132, 260, 1270],
        "description": "Sci-Fi with robots and space",
    },
    {
        "id": "test_05_horror_scary",
        "user_input": "I want a scary horror movie that's creepy and terrifying",
        "relevant_movie_ids": [1219, 1258, 1214, 2288, 1200],
        "description": "Scary horror film",
    },
    {
        "id": "test_06_romance_love",
        "user_input": "Can you suggest a romantic movie with love stories?",
        "relevant_movie_ids": [356, 1477, 909, 910, 2175],
        "description": "Romantic love story",
    },
    {
        "id": "test_07_animation",
        "user_input": "I want to watch an animated cartoon movie",
        "relevant_movie_ids": [5618, 31658, 3000, 364, 134853],
        "description": "Animated cartoon",
    },
    {
        "id": "test_08_year_constraint",
        "user_input": "Find me a movie from the 2000s that was popular",
        "relevant_movie_ids": [5618, 31658, 58559, 4226, 6016],
        "description": "Movie from 2000s",
    },
    {
        "id": "test_09_rating_high",
        "user_input": "I want a movie with a rating above 8.0, very highly rated",
        "relevant_movie_ids": [1870, 318, 858, 5618, 112552],
        "description": "Highly rated movie (>8.0)",
    },
    {
        "id": "test_10_multilingual",
        "user_input": "Find me a movie in spanish language, por favor",
        "relevant_movie_ids": [71033, 4235, 48394, 5319, 44694],
        "description": "Spanish language movie",
    },
]


def get_test_scenarios():
    return TEST_SCENARIOS


def get_test_scenario(test_id):
    return next((s for s in TEST_SCENARIOS if s["id"] == test_id), None)


def print_test_scenarios():
    print("=" * 70)
    print("TEST SCENARIOS FOR CINEASSIST EVALUATION")
    print("=" * 70)
    for scenario in TEST_SCENARIOS:
        print(f"\n[{scenario['id']}] {scenario['description']}")
        print(f"  User Input: {scenario['user_input']}")
        print(f"  Relevant Movie IDs: {scenario['relevant_movie_ids']}")
    print("\n" + "=" * 70)
