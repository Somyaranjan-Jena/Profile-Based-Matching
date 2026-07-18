import pandas as pd

from models.similarity import users, similarity_matrix
from models.mbti import mbti_score


SEMANTIC_WEIGHT = 0.50
MBTI_WEIGHT = 0.25
INTEREST_WEIGHT = 0.15
LOCATION_WEIGHT = 0.10


def interest_score(interests1, interests2):
    set1 = set(interests1.split(", "))
    set2 = set(interests2.split(", "))

    if not (set1 or set2):
        return 0

    return (len(set1 & set2) / len(set1 | set2)) * 100


def location_score(location1, location2):
    return 100 if location1 == location2 else 0


def calculate_match_score(user1, user2):

    semantic = similarity_matrix[user1.name][user2.name] * 100

    mbti = mbti_score(
        user1["mbti"],
        user2["mbti"]
    )

    interest = interest_score(
        user1["interests"],
        user2["interests"]
    )

    location = location_score(
        user1["location"],
        user2["location"]
    )

    compatibility = (
        semantic * SEMANTIC_WEIGHT +
        mbti * MBTI_WEIGHT +
        interest * INTEREST_WEIGHT +
        location * LOCATION_WEIGHT
    )

    return {
        "Semantic": round(float(semantic), 2),
        "MBTI": round(float(mbti), 2),
        "Interest": round(float(interest), 2),
        "Location": round(float(location), 2),
        "Compatibility": round(float(compatibility), 2),
    }


def recommend_matches(user_id, top_n=5):

    if user_id not in users["user_id"].values:
        raise ValueError(f"User '{user_id}' not found.")

    target = users[users["user_id"] == user_id].iloc[0]

    recommendations = []

    for _, candidate in users.iterrows():

        if candidate["user_id"] == user_id:
            continue

        scores = calculate_match_score(target, candidate)

        recommendations.append({
            "User ID": candidate["user_id"],
            "Name": candidate["name"],
            "Profession": candidate["profession"],
            "Location": candidate["location"],
            "MBTI Type": candidate["mbti"],
            "Semantic": scores["Semantic"],
            "MBTI": scores["MBTI"],
            "Interest": scores["Interest"],
            "Location Score": scores["Location"],
            "Compatibility": scores["Compatibility"],
        })

    recommendations = pd.DataFrame(recommendations)

    recommendations = recommendations.sort_values(
        "Compatibility",
        ascending=False
    )

    return recommendations.head(top_n).reset_index(drop=True)


if __name__ == "__main__":

    sample_user = users.iloc[0]["user_id"]

    print(f"\nTop Recommendations for {sample_user}\n")

    recommendations = recommend_matches(sample_user)

    print(recommendations)