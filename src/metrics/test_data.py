"""
Predefined test scenarios for evaluating the chatbot recommender.

Each test case contains:
- user_input: Natural language description of movie preferences
- relevant_movie_ids: List of movieIds that should be recommended (ground truth)
- description: Explanation of the test case
"""

TEST_SCENARIOS = [
    {
        "id": "test_01_action_movie",
        "user_input": "I want an action movie with lots of fights and explosions",
        "relevant_movie_ids": [79132, 58559, 72998, 89745, 122904],
        "description": "Action movie with fighting and explosions"
    },
    {
        "id": "test_02_comedy",
        "user_input": "recommend me a funny comedy movie that will make me laugh",
        "relevant_movie_ids": [122904, 356, 106782, 134853, 68954],
        "description": "Funny comedy for entertainment"
    },
    {
        "id": "test_03_drama_emotional",
        "user_input": "I'm looking for an emotional drama with touching storylines",
        "relevant_movie_ids": [109487, 58559, 2959, 356, 99114],
        "description": "Emotional drama film"
    },
    {
        "id": "test_04_scifi_future",
        "user_input": "Find me a sci-fi movie with robots and futuristic space themes",
        "relevant_movie_ids": [79132, 109487, 58559, 72998, 89745],
        "description": "Sci-Fi with robots and space"
    },
    {
        "id": "test_05_horror_scary",
        "user_input": "I want a scary horror movie that's creepy and terrifying",
        "relevant_movie_ids": [175303, 166534, 1258, 168250, 103249],
        "description": "Scary horror film"
    },
    {
        "id": "test_06_romance_love",
        "user_input": "Can you suggest a romantic movie with love stories?",
        "relevant_movie_ids": [356, 1721, 164909, 168366, 7361],
        "description": "Romantic love story"
    },
    {
        "id": "test_07_animation",
        "user_input": "I want to watch an animated cartoon movie",
        "relevant_movie_ids": [79132, 109487, 58559, 72998, 89745],
        "description": "Animated cartoon"
    },
    {
        "id": "test_08_year_constraint",
        "user_input": "Find me a movie from the 2000s that was popular",
        "relevant_movie_ids": [58559, 72998, 4896, 59315, 4993],
        "description": "Movie from 2000s"
    },
    {
        "id": "test_09_rating_high",
        "user_input": "I want a movie with a rating above 8.0, very highly rated",
        "relevant_movie_ids": [79132, 109487, 58559, 122912, 2959],
        "description": "Highly rated movie (>8.0)"
    },
    {
        "id": "test_10_multilingual",
        "user_input": "Find me a movie in spanish language, por favor",
        "relevant_movie_ids": [48394, 206246, 167832, 57274, 89118],
        "description": "Spanish language movie"
    },
]


def get_test_scenarios():
    """Return all test scenarios."""
    return TEST_SCENARIOS


def get_test_scenario(test_id):
    """
    Get a specific test scenario by ID.

    Args:
        test_id: The test scenario ID (e.g., "test_01_action_movie")

    Returns:
        dict: Test scenario or None if not found
    """
    for scenario in TEST_SCENARIOS:
        if scenario["id"] == test_id:
            return scenario
    return None


def print_test_scenarios():
    """Print all test scenarios in a readable format."""
    print("=" * 70)
    print("TEST SCENARIOS FOR CINEASSIST EVALUATION")
    print("=" * 70)

    for scenario in TEST_SCENARIOS:
        print(f"\n[{scenario['id']}] {scenario['description']}")
        print(f"  User Input: {scenario['user_input']}")
        print(f"  Relevant Movie IDs: {scenario['relevant_movie_ids']}")

    print("\n" + "=" * 70)
