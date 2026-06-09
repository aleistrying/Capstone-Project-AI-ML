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
        "relevant_movie_ids": [1, 2, 3, 5, 50],  # Action movies in dataset
        "description": "Action movie with fighting and explosions"
    },
    {
        "id": "test_02_comedy",
        "user_input": "recommend me a funny comedy movie that will make me laugh",
        "relevant_movie_ids": [10, 15, 20, 25, 30],  # Comedy movies in dataset
        "description": "Funny comedy for entertainment"
    },
    {
        "id": "test_03_drama_emotional",
        "user_input": "I'm looking for an emotional drama with touching storylines",
        "relevant_movie_ids": [40, 42, 45, 48, 52],  # Drama movies in dataset
        "description": "Emotional drama film"
    },
    {
        "id": "test_04_scifi_future",
        "user_input": "Find me a sci-fi movie with robots and futuristic space themes",
        "relevant_movie_ids": [60, 65, 70, 75, 80],  # Sci-Fi movies in dataset
        "description": "Sci-Fi with robots and space"
    },
    {
        "id": "test_05_horror_scary",
        "user_input": "I want a scary horror movie that's creepy and terrifying",
        "relevant_movie_ids": [100, 105, 110, 115, 120],  # Horror movies in dataset
        "description": "Scary horror film"
    },
    {
        "id": "test_06_romance_love",
        "user_input": "Can you suggest a romantic movie with love stories?",
        "relevant_movie_ids": [130, 135, 140, 145, 150],  # Romance movies in dataset
        "description": "Romantic love story"
    },
    {
        "id": "test_07_animation",
        "user_input": "I want to watch an animated cartoon movie",
        "relevant_movie_ids": [160, 165, 170, 175, 180],  # Animation movies in dataset
        "description": "Animated cartoon"
    },
    {
        "id": "test_08_year_constraint",
        "user_input": "Find me a movie from the 2000s that was popular",
        "relevant_movie_ids": [200, 205, 210, 215, 220],  # Movies from 2000-2009
        "description": "Movie from 2000s"
    },
    {
        "id": "test_09_rating_high",
        "user_input": "I want a movie with a rating above 8.0, very highly rated",
        "relevant_movie_ids": [250, 255, 260, 265, 270],  # High-rated movies (>8.0)
        "description": "Highly rated movie (>8.0)"
    },
    {
        "id": "test_10_multilingual",
        "user_input": "Find me a movie in spanish language, por favor",
        "relevant_movie_ids": [300, 305, 310, 315, 320],  # Spanish language movies
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
