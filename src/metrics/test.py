from recommender_engine import recommend_on_the_fly
import sys
import os
sys.path.append(os.path.abspath(
    '/Workspace/Users/carlos.graniel.manrique@gmail.com/Capstone-Project-AI-ML/src/nlp'))
# sys.path.append(os.path.abspath(
#    '/Workspace/Users/carlos.graniel.manrique@gmail.com/Capstone-Project-AI-ML/src/recommender'))


preferences = extract_preferences(
    "I'm in the mood for a funny comedy movie in english with high ratings")
display(preferences)
# recomm = recommend_on_the_fly(preferences)
# display(
