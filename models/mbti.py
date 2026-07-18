MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]


def mbti_score(type1, type2):
    if type1 == type2:
        return 90

    score = 50

    for a, b in zip(type1, type2):

        if a == b:
            score += 10
        else:
            score += 5

    return min(score, 100)


if __name__ == "__main__":

    print("INTJ vs ENFP :", mbti_score("INTJ", "ENFP"))
    print("INTJ vs INTJ :", mbti_score("INTJ", "INTJ"))
    print("ENFP vs INFJ :", mbti_score("ENFP", "INFJ"))