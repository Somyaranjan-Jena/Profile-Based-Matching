import os
import random
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
Faker.seed(42)

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

LOCATIONS = [
    "Bangalore", "Hyderabad", "Chennai", "Mumbai",
    "Delhi", "Pune", "Kolkata", "Ahmedabad",
    "Bhubaneswar", "Noida", "Gurgaon", "Kochi"
]

PROFESSIONS = {
    "Software Engineer": {
        "skills": ["Python", "Java", "Docker", "AWS", "Microservices", "Git"],
        "interests": ["Coding", "Gaming", "Open Source", "AI"]
    },
    "Data Scientist": {
        "skills": ["Python", "SQL", "Machine Learning", "NLP", "TensorFlow", "Statistics"],
        "interests": ["Analytics", "AI", "Reading", "Research"]
    },
    "Business Analyst": {
        "skills": ["Excel", "SQL", "Power BI", "Communication"],
        "interests": ["Finance", "Travel", "Reading", "Business"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Linux", "Network Security", "SIEM", "Penetration Testing"],
        "interests": ["Technology", "Gaming", "CTF", "Reading"]
    },
    "DevOps Engineer": {
        "skills": ["Docker", "Kubernetes", "AWS", "CI/CD"],
        "interests": ["Automation", "Cloud", "Linux", "Coding"]
    },
    "Doctor": {
        "skills": ["Diagnosis", "Patient Care", "Healthcare", "Medicine"],
        "interests": ["Fitness", "Reading", "Travel", "Volunteering"]
    },
    "Teacher": {
        "skills": ["Teaching", "Mentoring", "Communication", "Curriculum Design"],
        "interests": ["Books", "Writing", "Music", "Education"]
    },
    "UI UX Designer": {
        "skills": ["Figma", "Wireframing", "Prototyping", "User Research"],
        "interests": ["Design", "Photography", "Art", "Travel"]
    }
}

SUMMARY_TEMPLATES = [
    "Experienced {profession} with expertise in {skill1} and {skill2}. Passionate about solving practical problems and continuously improving technical skills.",
    "{profession} focused on delivering quality solutions using {skill1}. Enjoys collaborative environments and innovative projects involving {skill2}.",
    "Dedicated {profession} with hands-on experience in {skill1}. Interested in building scalable solutions while expanding expertise in {skill2}.",
]

ABOUT_TEMPLATES = [
    "I enjoy {interest1}, {interest2}, and working with collaborative teams. I value integrity, creativity, and continuous learning.",
    "My work style emphasizes teamwork, problem solving, and innovation. Outside work I enjoy {interest1} and {interest2}.",
    "I believe in lifelong learning, helping others grow, and maintaining a positive work environment. My hobbies include {interest1} and {interest2}.",
]


def mbti_score(a, b):
    if a == b:
        return 90

    score = 50

    for x, y in zip(a, b):
        if x == y:
            score += 10
        else:
            score += 5

    return min(score, 100)


users = []

for i in range(1, 101):
    profession = random.choice(list(PROFESSIONS.keys()))
    info = PROFESSIONS[profession]

    skill1, skill2 = random.sample(info["skills"], 2)
    interest1, interest2 = random.sample(info["interests"], 2)

    summary = random.choice(SUMMARY_TEMPLATES).format(
        profession=profession,
        skill1=skill1,
        skill2=skill2
    )

    about = random.choice(ABOUT_TEMPLATES).format(
        interest1=interest1.lower(),
        interest2=interest2.lower()
    )

    users.append({
        "user_id": f"U{i:03}",
        "name": fake.name(),
        "age": random.randint(21, 45),
        "location": random.choice(LOCATIONS),
        "profession": profession,
        "experience_years": random.randint(0, 20),
        "professional_summary": summary,
        "about_me": about,
        "mbti": random.choice(MBTI_TYPES),
        "interests": f"{interest1}, {interest2}"
    })

users_df = pd.DataFrame(users)

feedback = []

for _, user in users_df.iterrows():

    possible = users_df[users_df["user_id"] != user["user_id"]]

    matches = possible.sample(random.randint(5, 10), random_state=random.randint(1, 10000))

    for _, other in matches.iterrows():

        score = 0

        if user["profession"] == other["profession"]:
            score += 40

        if user["location"] == other["location"]:
            score += 20

        score += mbti_score(user["mbti"], other["mbti"]) * 0.4

        interest_overlap = len(
            set(user["interests"].split(", ")) &
            set(other["interests"].split(", "))
        )

        score += interest_overlap * 20

        score = min(score, 100)

        if score >= 85:
            action = random.choices([1, 0], weights=[95, 5])[0]
        elif score >= 70:
            action = random.choices([1, 0], weights=[80, 20])[0]
        elif score >= 55:
            action = random.choices([1, 0], weights=[55, 45])[0]
        else:
            action = random.choices([1, 0], weights=[15, 85])[0]

        feedback.append({
            "user_id": user["user_id"],
            "matched_user_id": other["user_id"],
            "action": action,
            "timestamp": (
                datetime.now() -
                timedelta(days=random.randint(0, 365))
            ).strftime("%Y-%m-%d")
        })

feedback_df = pd.DataFrame(feedback)

os.makedirs("data", exist_ok=True)

users_df.to_csv("data/users.csv", index=False)
feedback_df.to_csv("data/feedback.csv", index=False)

print("=" * 60)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"Users          : {len(users_df)}")
print(f"Feedback Rows  : {len(feedback_df)}")
print(f"Acceptance Rate: {feedback_df['action'].mean()*100:.2f}%")
print("=" * 60)